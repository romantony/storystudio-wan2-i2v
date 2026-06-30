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
    sed -i 's/device=torch\.cuda\.current_device()/device=0/g' /workspace/wan22/wan/modules/t5.py && \
    # FlashAttention is not installed (does not build on these images). Route every
    # `flash_attention` import to attention(), which has a PyTorch SDPA fallback.
    grep -rl 'import flash_attention' /workspace/wan22/wan | \
    xargs -r sed -i \
      -e 's/from \.attention import flash_attention/from .attention import attention as flash_attention/g' \
      -e 's/from \.\.modules\.attention import flash_attention/from ..modules.attention import attention as flash_attention/g' \
      -e 's/from wan\.modules\.attention import flash_attention/from wan.modules.attention import attention as flash_attention/g'

# Pin torch + torchvision to a CUDA 12.8 build. cu128 supports BOTH Blackwell
# (RTX 5090) and Ada (RTX 6000 Ada / L40S), and runs on any host whose driver
# supports CUDA >= 12.8 (incl. the 12.9 Ada hosts). Installing torchvision
# unpinned would otherwise drag in a cu130 torch that needs a CUDA-13 driver
# many RunPod hosts don't have, crashing torch.cuda init with "driver too old".
RUN python3 -m pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip cache purge

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

# Verify critical packages import and confirm the torch CUDA build is 12.x (not 13.x)
RUN python3 -c "import runpod, diffusers, torch, torchvision, easydict; \
print(f'OK — runpod={runpod.__version__} diffusers={diffusers.__version__} torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.version.cuda}'); \
assert torch.version.cuda.startswith('12'), f'torch CUDA build {torch.version.cuda} requires too-new a driver'"

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
