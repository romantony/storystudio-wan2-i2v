#!/usr/bin/env python
"""
Download the official Wan2.2-I2V BF16 model from HuggingFace.

Usage:
    python3 scripts/download_wan22_bf16.py \
        --repo_id  Wan-Video/Wan2.2-I2V-14B \
        --local_dir /workspace/models/wan22-i2v-bf16

Find the correct repo_id: https://huggingface.co/Wan-Video
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id',   required=True,
                        help='HuggingFace repo ID, e.g. Wan-Video/Wan2.2-I2V-14B')
    parser.add_argument('--local_dir', required=True,
                        help='Local path to download into')
    parser.add_argument('--token',     default=None,
                        help='HF token if the repo is gated (optional)')
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} → {local_dir}")
    print("(This will take a while — ~60 GB for both DiTs + T5 + VAE)")

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        repo_type="model",
        token=args.token,
    )

    print(f"\n✓ Downloaded to {path}")

if __name__ == '__main__':
    main()
