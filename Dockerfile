# Wan2.2-I2V-A14B-FP8 RunPod Serverless Dockerfile
# Optimised for RTX 5090 (Blackwell) — requires CUDA 12.6+ and PyTorch 2.6+
FROM runpod/pytorch:1.0.7-cu1290-torch260-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/workspace/models/wan22-i2v-fp8 \
    HF_HOME=/workspace/huggingface \
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

# Clone Wan2.2 native code — provides wan.image2video.WanI2V and model classes
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /workspace/wan22

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
    requests==2.32.3 \
    boto3==1.35.76 \
    runpod==1.7.5 \
    filelock \
    "packaging>=20.0" \
    tqdm && \
    python3 -m pip cache purge

# Verify critical packages are importable by the runtime Python
RUN python3 -c "import runpod; import diffusers; import torch; import easydict; print(f'OK — runpod={runpod.__version__} diffusers={diffusers.__version__} torch={torch.__version__}')"

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
