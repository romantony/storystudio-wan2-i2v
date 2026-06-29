#!/usr/bin/env python3
"""
Quantize Wan2.2-I2V-A14B from BF16 to our FP8 format.

Loads directly from the downloaded safetensors shards — no WanI2V, no GPU needed.
Processes one input shard at a time so peak RAM is ~one shard (~10 GB) plus the
accumulated FP8 output for the current DiT (~28 GB). Works on CPU only.

Usage:
    python3 quantize_fp8.py
    python3 quantize_fp8.py /workspace/data/wan22-bf16 /workspace/data/wan22-qfp8

Output
------
{OUTPUT_DIR}/
  high_noise_model/   blocks.0.safetensors … head.safetensors …
  low_noise_model/    (same)
  models_t5_umt5-xxl-enc-bf16.pth   copied
  Wan2.1_VAE.pth                     copied
  google/                            copied (tokenizer)
  configuration.json                 copied
  format.json                        written by this script
"""
import gc
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

BF16_DIR   = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/wan22-bf16"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "/workspace/data/wan22-qfp8"

import torch
from safetensors.torch import load_file, save_file


# ── Key helpers ───────────────────────────────────────────────────────────────

def _top_group(key: str) -> str:
    """Map a state-dict key to the output shard file stem.

    blocks.0.self_attn.q.weight  →  blocks.0
    head.head.weight             →  head
    patch_embedding.weight       →  patch_embedding
    time_embedding.0.weight      →  time_embedding
    """
    parts = key.split('.')
    if parts[0] == 'blocks':
        return f"blocks.{parts[1]}"
    return parts[0]


def _is_linear_weight(key: str, tensor: torch.Tensor) -> bool:
    """True for nn.Linear weight matrices (2-D, not norm/embed/conv)."""
    if tensor.dim() != 2 or not key.endswith('.weight'):
        return False
    lower = key.lower()
    return not any(t in lower for t in ('norm', 'embed', 'patch_embed', 'pos_embed'))


def _quantize(w: torch.Tensor):
    """Per-tensor absmax FP8.  Returns (fp8_weight, bf16_scale)."""
    w_f32 = w.float()
    scale = (w_f32.abs().max() / 448.0).clamp(min=1e-12)
    fp8_w = (w_f32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return fp8_w, scale.to(torch.bfloat16)


# ── Per-DiT quantization ──────────────────────────────────────────────────────

def _quantize_dit(dit_dir: Path, out_dir: Path) -> None:
    """
    Read sharded safetensors from dit_dir, quantize linear weights to FP8,
    save as per-block safetensors to out_dir. CPU-only, one shard at a time.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    index_file = dit_dir / 'diffusion_pytorch_model.safetensors.index.json'
    if not index_file.exists():
        # Single-file model (no sharding)
        shard_files = sorted(dit_dir.glob('diffusion_pytorch_model*.safetensors'))
        weight_map  = {None: shard_files}   # sentinel: load all keys from each file
    else:
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index['weight_map']    # key → shard filename

    # Build: shard filename → list of keys in that shard
    shard_to_keys: dict[str, list] = defaultdict(list)
    for key, shard in weight_map.items():
        shard_to_keys[shard].append(key)

    # Accumulate output groups across all shards
    output_groups: dict[str, dict] = defaultdict(dict)
    n_total = n_fp8 = 0

    for shard_name in sorted(shard_to_keys):
        shard_path = dit_dir / shard_name
        print(f"  {shard_name} ...", end='', flush=True)
        t0 = time.time()

        tensors = load_file(str(shard_path))

        for key in shard_to_keys[shard_name]:
            tensor = tensors[key]
            group  = _top_group(key)

            if _is_linear_weight(key, tensor):
                fp8_w, scale = _quantize(tensor)
                output_groups[group][key]            = fp8_w
                output_groups[group][f"{key}.scale"] = scale
                n_fp8 += 1
            else:
                output_groups[group][key] = tensor.to(torch.bfloat16)

            n_total += 1

        del tensors
        gc.collect()
        print(f" {time.time()-t0:.1f}s")

    # Save per-block files
    for group_name in sorted(output_groups):
        save_file(output_groups[group_name], str(out_dir / f"{group_name}.safetensors"))

    print(f"  → {n_total} params: {n_fp8} FP8 + {n_total-n_fp8} BF16"
          f"  |  {len(output_groups)} shard files saved")
    del output_groups
    gc.collect()


# ── Copy non-DiT assets ───────────────────────────────────────────────────────

def _copy_assets(bf16_dir: Path, out_dir: Path) -> None:
    """Copy T5, VAE, tokenizer and config files unchanged."""
    COPY_DIRS  = ('google',)                 # tokenizer (google/umt5-xxl/)
    COPY_FILES = (
        'models_t5_umt5-xxl-enc-bf16.pth',  # T5 encoder weights
        'Wan2.1_VAE.pth',                    # VAE weights
        'configuration.json',
        'config.json',
    )

    for name in COPY_DIRS:
        src = bf16_dir / name
        if src.exists():
            dst = out_dir / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {name}/")

    for name in COPY_FILES:
        src = bf16_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            print(f"  Copied {name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    bf16_dir   = Path(BF16_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not bf16_dir.exists():
        sys.exit(f"BF16 model not found: {bf16_dir}\nRun download.py first.")

    print(f"PyTorch {torch.__version__}")
    print(f"BF16:  {bf16_dir}")
    print(f"Out:   {output_dir}")
    print("Mode:  CPU-only (no GPU needed for weight quantization)\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    for subdir in ('high_noise_model', 'low_noise_model'):
        dit_dir = bf16_dir / subdir
        if not dit_dir.exists():
            print(f"WARNING: {dit_dir} not found, skipping")
            continue
        print(f"Quantizing {subdir} ...")
        t0 = time.time()
        _quantize_dit(dit_dir, output_dir / subdir)
        print(f"  Done in {time.time()-t0:.0f}s\n")

    print("Copying T5 / VAE / tokenizer ...")
    _copy_assets(bf16_dir, output_dir)

    fmt = {
        "format":         "wan22-i2v-qfp8-v1",
        "quant":          "float8_e4m3fn",
        "scale":          "per_tensor_absmax",
        "non_linear":     "bfloat16",
        "high_noise_dir": "high_noise_model",
        "low_noise_dir":  "low_noise_model",
    }
    (output_dir / 'format.json').write_text(json.dumps(fmt, indent=2))

    print(f"\n✓ Done → {output_dir}")
    print("Copy this directory to your production network volume,")
    print("then set MODEL_PATH to point at it.")


if __name__ == '__main__':
    main()
