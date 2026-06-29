#!/usr/bin/env python
"""
Quantize Wan2.2-I2V-A14B from BF16 to our own FP8 format.

Run ONCE on an RTX Pro 6000 (96 GB VRAM) pod:
  python3 quantize_wan22_fp8.py --bf16_dir /workspace/models/wan22-i2v-bf16
                                --output_dir /workspace/models/wan22-i2v-qfp8

What this does
--------------
1. Loads both DiTs as BF16 (both fit in 96 GB VRAM simultaneously).
2. For every nn.Linear weight matrix:
     scale = max(|weight|) / 448          (448 = max value of float8_e4m3fn)
     fp8_w = (weight / scale).to(float8_e4m3fn)
3. Saves per-block safetensors with ABSOLUTE keys:
     blocks.0.self_attn.q.weight       → float8_e4m3fn
     blocks.0.self_attn.q.weight.scale → bfloat16 scalar
     blocks.0.self_attn.q.bias         → bfloat16
     blocks.0.norm1.weight             → bfloat16   (non-linear: not quantized)
     ...
4. Non-linear params (norms, Conv3d patch embeddings, time/text embeddings) stay BF16.
5. T5, VAE, tokenizer: copied from the BF16 source directory unchanged.

Output layout (per DiT)
-----------------------
{output_dir}/
  high_noise_model/
    blocks.0.safetensors
    blocks.1.safetensors
    ...
    blocks.39.safetensors
    head.safetensors
    patch_embedding.safetensors
    text_embedding.safetensors
    time_embedding.safetensors
    time_projection.safetensors
  low_noise_model/
    (same)
  format.json          ← marks this as our FP8 format for the model_server
  text_encoder/        ← copied from bf16_dir
  tokenizer/           ← copied from bf16_dir
  vae/                 ← copied from bf16_dir
  model_index.json     ← copied from bf16_dir (if present)
"""

import argparse
import gc
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _top_group(key: str) -> str:
    """Map an absolute state-dict key to the file stem for its shard file.

    Examples
      blocks.0.self_attn.q.weight → blocks.0
      head.head.weight            → head
      patch_embedding.weight      → patch_embedding
      time_embedding.0.weight     → time_embedding
    """
    parts = key.split('.')
    if parts[0] == 'blocks':
        return f"blocks.{parts[1]}"
    return parts[0]


def _is_linear_weight(key: str, tensor: torch.Tensor) -> bool:
    """Return True if this tensor is a quantisable nn.Linear weight."""
    if tensor.dim() != 2:
        return False
    if not key.endswith('.weight'):
        return False
    # Skip embeddings and norms (they are 2-D but not linear projection weights)
    lower = key.lower()
    skip_tokens = ('norm', 'embed', 'patch_embedding', 'pos_embed', 'ln_')
    return not any(tok in lower for tok in skip_tokens)


# ---------------------------------------------------------------------------
# Per-block quantisation and save
# ---------------------------------------------------------------------------

def _quantize_and_save_dit(model: nn.Module, output_dir: Path) -> None:
    """Walk a WanModel state dict, quantize linear weights, save per-block files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()

    # Bucket tensors by shard group
    groups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    linear_weight_keys: set[str] = set()

    for key, tensor in state_dict.items():
        group = _top_group(key)
        if _is_linear_weight(key, tensor):
            linear_weight_keys.add(key)
            fp8_w, scale = _quantize_tensor(tensor)
            groups[group][key]            = fp8_w
            groups[group][f"{key}.scale"] = scale
        else:
            groups[group][key] = tensor.to(torch.bfloat16)

    n_linear  = len(linear_weight_keys)
    n_other   = len(state_dict) - n_linear
    n_total   = len(state_dict)
    print(f"    {n_total} params: {n_linear} linear → FP8, {n_other} other → BF16")

    from safetensors.torch import save_file
    for group_name in sorted(groups):
        tensors = groups[group_name]
        out_file = output_dir / f"{group_name}.safetensors"
        save_file(tensors, str(out_file))

    print(f"    Saved {len(groups)} shard files → {output_dir}")


def _quantize_tensor(w: torch.Tensor):
    """Per-tensor absmax FP8 quantisation.  Returns (fp8_weight, scale)."""
    w_f32 = w.float()
    scale  = w_f32.abs().max() / 448.0          # 448 = max of float8_e4m3fn
    scale  = scale.clamp(min=1e-12)              # avoid div-by-zero
    fp8_w  = (w_f32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return fp8_w, scale.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quantize Wan2.2-I2V DiTs to FP8")
    parser.add_argument('--bf16_dir',    required=True, help="Path to downloaded BF16 checkpoint")
    parser.add_argument('--output_dir',  required=True, help="Output path for our FP8 format")
    parser.add_argument('--wan22_src',   default='/workspace/wan22',
                        help="Path to cloned Wan-Video/Wan2.2 repo")
    parser.add_argument('--device_id',   type=int, default=0)
    args = parser.parse_args()

    bf16_dir   = Path(args.bf16_dir)
    output_dir = Path(args.output_dir)

    if not bf16_dir.exists():
        sys.exit(f"BF16 checkpoint not found: {bf16_dir}")

    sys.path.insert(0, args.wan22_src)
    from wan import WanI2V
    from wan.configs import WAN_CONFIGS

    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    gpu_name = torch.cuda.get_device_name(args.device_id)
    vram_gb  = torch.cuda.get_device_properties(args.device_id).total_memory / 1024**3
    print(f"GPU: {gpu_name}  |  VRAM: {vram_gb:.1f} GB")

    # -----------------------------------------------------------------------
    # Load BF16 model
    # -----------------------------------------------------------------------
    print(f"\nLoading BF16 model from {bf16_dir} ...")

    import logging
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    cfg = WAN_CONFIGS['i2v-A14B']

    pipe = WanI2V(
        config=cfg,
        checkpoint_dir=str(bf16_dir),
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=False,       # load directly to GPU — 96 GB VRAM fits both DiTs
        convert_model_dtype=False,
    )

    # -----------------------------------------------------------------------
    # Quantise each DiT
    # -----------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    for attr, subdir in [
        ('high_noise_model', 'high_noise_model'),
        ('low_noise_model',  'low_noise_model'),
    ]:
        model = getattr(pipe, attr).cpu()       # move to CPU for state_dict access
        print(f"\nQuantizing {attr} ...")
        _quantize_and_save_dit(model, output_dir / subdir)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # -----------------------------------------------------------------------
    # Copy non-DiT assets unchanged (T5, VAE, tokenizer, config files)
    # -----------------------------------------------------------------------
    print("\nCopying T5 / VAE / tokenizer ...")
    COPY_DIRS = ('text_encoder', 'tokenizer', 'vae', 'scheduler')
    COPY_FILES = ('model_index.json', 'config.json')

    for name in COPY_DIRS:
        src = bf16_dir / name
        if src.exists():
            dst = output_dir / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {name}/")

    for name in COPY_FILES:
        src = bf16_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            print(f"  Copied {name}")

    # -----------------------------------------------------------------------
    # Write format marker — model_server.py uses this to pick the right loader
    # -----------------------------------------------------------------------
    fmt = {
        "format": "wan22-i2v-qfp8-v1",
        "quant": "float8_e4m3fn",
        "scale": "per_tensor_absmax",
        "non_linear": "bfloat16",
        "high_noise_dir": "high_noise_model",
        "low_noise_dir": "low_noise_model",
    }
    (output_dir / 'format.json').write_text(json.dumps(fmt, indent=2))

    print(f"\n✓ Quantisation complete → {output_dir}")
    print("  Transfer the output_dir to your production network volume and")
    print("  point MODEL_PATH to it.  model_server.py detects format.json automatically.")


if __name__ == '__main__':
    main()
