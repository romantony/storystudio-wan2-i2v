# Wan2.2-I2V-A14B-FP8 RunPod Serverless Dockerfile
# Optimised for RTX 5090 (Blackwell) — requires CUDA 12.6+ and PyTorch 2.6+
FROM runpod/pytorch:1.0.7-cu1290-torch260-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/runpod-volume/wan22-qfp8 \
    HF_HOME=/runpod-volume/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /workspace

# Clone Wan2.2 native code — provides wan.image2video.WanI2V and model classes.
# Rewrite wan/__init__.py to only import what we need; the original eagerly imports
# WanS2V (needs decord), WanT2V, WanTI2V, and WanAnimate — none used here.
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /workspace/wan22 && \
    printf '# trimmed for i2v-only usage\nfrom . import configs, distributed, modules\nfrom .image2video import WanI2V\n' \
    > /workspace/wan22/wan/__init__.py && \
    sed -i 's/device=torch\.cuda\.current_device()/device=0/g' /workspace/wan22/wan/modules/t5.py

# Patch rope_apply to use float32 instead of float64 internally.
#
# rope_apply uses torch.float64 for numerical stability, but on RTX 6000 Ada (47.4 GB)
# with both 14B FP8 DiTs resident (~45.5 GB base VRAM), the float64 path peaks at:
#
#   x[i, :seq_len].to(float64)           →  1.30 GB complex128 intermediate
#   x_i * freqs_i (complex128 result)    →  +1.30 GB  (simultaneous with source)
#   view_as_real → flatten → output list →  1.30 GB float64 in output
#   torch.stack(output).float()          →  +0.62 GB float32  ← OOM (only 602 MB free)
#
# Fix: use float32 (.float()) instead of float64 (.to(torch.float64)), and cast freqs_i
# to complex64 so the multiply stays in float32 instead of upcasting to float64.
# Peak with float32: two simultaneous complex64 copies = 2×0.65 GB = 1.30 GB above base
# (vs 2×1.30 GB = 2.60 GB with float64). float32 RoPE is standard in all other models.
RUN python3 - << 'PYEOF'
import pathlib, re

path = pathlib.Path('/workspace/wan22/wan/modules/model.py')
code = path.read_text()

# 1. Downgrade intermediate precision: float64 → float32
old1 = '.to(torch.float64).reshape('
new1 = '.float().reshape('
if old1 not in code:
    raise RuntimeError(f"Patch target 1 not found in {path}")
code = code.replace(old1, new1, 1)

# 2. Cast freqs_i to match x_i dtype (complex64) to prevent upcast during multiply.
#    Without this, complex64 × complex128 → complex128 (no memory savings).
old2 = 'x_i = torch.view_as_real(x_i * freqs_i).flatten(2)'
new2 = 'x_i = torch.view_as_real(x_i * freqs_i.to(x_i.dtype)).flatten(2)'
if old2 not in code:
    raise RuntimeError(f"Patch target 2 not found in {path}")
code = code.replace(old2, new2, 1)

path.write_text(code)
print(f"Patched rope_apply in {path}: float64 → float32 (saves ~1.3 GB peak VRAM per call)")
PYEOF

# Pin torch + torchvision to a CUDA 12.8 build. cu128 supports BOTH Blackwell
# (RTX 5090) and Ada (RTX 6000 Ada / L40S), and runs on any host whose driver
# supports CUDA >= 12.8 (incl. the 12.9 Ada hosts). Installing torchvision
# unpinned would otherwise drag in a cu130 torch that needs a CUDA-13 driver
# many RunPod hosts don't have, crashing torch.cuda init with "driver too old".
RUN python3 -m pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip cache purge

# FlashAttention 2 (best-effort). Wan's attention routes to the real flash kernel
# when `flash_attn` imports; otherwise it falls back to PyTorch SDPA (the
# 'Padding mask disabled' path). Try prebuilt wheels matching torch 2.7 / cu12 /
# cp310 across both C++ ABIs and a few versions. NON-FATAL: a wheel mismatch must
# not break the build — we confirm activation from the startup log and pin the
# exact wheel later if needed.
RUN set +e; \
    for FA in 2.8.2 2.8.1 2.8.0.post2 2.7.4.post1; do \
      for ABI in TRUE FALSE; do \
        URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA}/flash_attn-${FA}+cu12torch2.7cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl"; \
        echo "Trying flash-attn $FA abi=$ABI"; \
        python3 -m pip install --no-cache-dir "$URL" && break 2; \
      done; \
    done; \
    python3 -c "import flash_attn" 2>/dev/null || \
      python3 -m pip install --no-cache-dir flash_attn --prefer-binary --no-build-isolation 2>&1 | tail -3; \
    python3 -c "import flash_attn; print('flash-attn', flash_attn.__version__)" \
      || echo "flash-attn NOT installed — runtime will use PyTorch SDPA fallback"; \
    python3 -m pip cache purge; true

# Redirect flash_attention imports to SDPA fallback ONLY when flash_attn is unavailable.
# If flash_attn installed successfully above, native flash_attention() uses the FA2 kernel
# and this step is a no-op. If not, redirect to attention() which falls back to PyTorch SDPA.
RUN python3 - << 'PYEOF'
import sys
try:
    import flash_attn
    print(f"flash_attn {flash_attn.__version__} present — native FA2 kernel active, skipping redirect")
    sys.exit(0)
except ImportError:
    pass
import pathlib
subs = [
    ('from .attention import flash_attention',            'from .attention import attention as flash_attention'),
    ('from ..modules.attention import flash_attention',   'from ..modules.attention import attention as flash_attention'),
    ('from wan.modules.attention import flash_attention', 'from wan.modules.attention import attention as flash_attention'),
]
patched = 0
for py in pathlib.Path('/workspace/wan22/wan').rglob('*.py'):
    try:
        code = py.read_text()
    except Exception:
        continue
    updated = code
    for old, new in subs:
        updated = updated.replace(old, new)
    if updated != code:
        py.write_text(updated)
        print(f"  patched {py}")
        patched += 1
print(f"flash_attn unavailable — redirected flash_attention→attention() in {patched} file(s) (SDPA fallback)")
PYEOF

RUN python3 -m pip install --no-cache-dir \
    transformers==4.51.3 \
    "diffusers>=0.33.0" \
    accelerate==1.3.0 \
    safetensors==0.4.5 \
    tokenizers==0.21.0 \
    sentencepiece==0.2.0 \
    huggingface-hub==0.30.0 \
    imageio==2.36.1 \
    imageio-ffmpeg==0.5.1 \
    pillow==11.0.0 \
    "numpy>=1.23.5,<2" \
    ftfy==6.3.1 \
    easydict \
    einops \
    regex \
    requests==2.32.3 \
    boto3==1.35.76 \
    runpod==1.7.5 \
    filelock \
    "packaging>=20.0" \
    tqdm && \
    python3 -m pip cache purge

# Verify critical packages import and confirm the torch CUDA build is 12.x (not 13.x).
# Also print torch's C++ ABI + flash-attn status so we can pin the right wheel if missed.
RUN python3 -c "import runpod, diffusers, torch, torchvision, easydict; \
print(f'OK — runpod={runpod.__version__} diffusers={diffusers.__version__} torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.version.cuda}'); \
print(f'torch cxx11_abi={torch._C._GLIBCXX_USE_CXX11_ABI}'); \
assert torch.version.cuda.startswith('12'), f'torch CUDA build {torch.version.cuda} requires too-new a driver'" && \
    (python3 -c "import flash_attn; print('flash-attn', flash_attn.__version__, 'OK')" \
     || echo "flash-attn not present — SDPA fallback")

# Verify Wan2.2 code was cloned correctly
RUN test -f /workspace/wan22/wan/configs/__init__.py && \
    test -f /workspace/wan22/wan/image2video.py && \
    echo "Wan2.2 source OK"

COPY handler_v2.py ./handler.py
COPY model_server.py ./handler/model_server.py

RUN mkdir -p /workspace/models /workspace/huggingface /workspace/outputs /workspace/handler

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "print('healthy')" || exit 1

CMD ["python3", "-u", "handler.py"]
