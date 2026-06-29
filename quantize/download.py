#!/usr/bin/env python3
"""
Download the official Wan2.2-I2V BF16 model from HuggingFace.

Usage:
    python3 download.py                                  # uses defaults below
    python3 download.py Wan-Video/Wan2.2-I2V-14B        # custom repo
    python3 download.py Wan-Video/Wan2.2-I2V-14B /data/models/wan22-bf16

Find the correct repo_id at: https://huggingface.co/Wan-Video
"""
import sys
from pathlib import Path

REPO_ID   = "Wan-Video/Wan2.2-I2V-14B"
LOCAL_DIR = "/workspace/models/wan22-i2v-bf16"

repo_id   = sys.argv[1] if len(sys.argv) > 1 else REPO_ID
local_dir = sys.argv[2] if len(sys.argv) > 2 else LOCAL_DIR

Path(local_dir).mkdir(parents=True, exist_ok=True)

print(f"Repo:      {repo_id}")
print(f"Dest:      {local_dir}")
print("Size:      ~60 GB — this will take a while\n")

from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    repo_type="model",
)

print(f"\n✓ Downloaded to {path}")
