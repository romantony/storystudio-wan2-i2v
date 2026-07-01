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

# Patch rope_apply to free the float64 intermediate list before the float32 conversion.
#
# rope_apply uses float64 internally for RoPE numerical stability. On a 48 GB GPU with
# the Wan2.2-14B dual-DiT setup (~44.7 GB base VRAM), the peak allocation sequence is:
#   float64 output list  →  1.34 GB
#   torch.stack(output)  →  +1.34 GB  (47.4 GB total — fits)
#   .float()             →  +0.62 GB  (48.0 GB — OOM!)
#
# Fix: rebind `output = torch.stack(output)`. CPython's reference counting immediately
# frees the float64 list when the name is rebound, so .float() sees only 46.1+0.62=46.7 GB.
RUN python3 - << 'PYEOF'
import pathlib
path = pathlib.Path('/workspace/wan22/wan/modules/model.py')
code = path.read_text()
old = '    return torch.stack(output).float()'
new = (
    '    output = torch.stack(output)  '
    '# rebind frees float64 list → .float() fits in 48 GB VRAM\n'
    '    return output.float()'
)
if old not in code:
    raise RuntimeError(f"rope_apply patch target not found in {path}; check Wan2.2 version")
patched = code.replace(old, new, 1)
path.write_text(patched)
print(f"Patched rope_apply in {path}: float64 list freed before float32 conversion")
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
