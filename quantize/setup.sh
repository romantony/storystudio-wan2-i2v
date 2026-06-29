#!/usr/bin/env bash
# Install Wan2.2 dependencies on the quantization pod.
# flash_attn is skipped — it requires a build env that can't find torch via pip.
set -e

WAN22_DIR="${WAN22_DIR:-/workspace/wan22}"

if [ ! -f "$WAN22_DIR/requirements.txt" ]; then
    echo "Wan2.2 source not found at $WAN22_DIR"
    echo "Run: git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git $WAN22_DIR"
    exit 1
fi

echo "Installing deps from $WAN22_DIR/requirements.txt (skipping flash_attn) ..."
grep -v flash_attn "$WAN22_DIR/requirements.txt" | pip install -r /dev/stdin

echo "Installing safetensors ..."
pip install safetensors

echo "✓ Setup complete"
