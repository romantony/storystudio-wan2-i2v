#!/usr/bin/env python3
"""
Quantize Wan2.2-I2V-A14B from BF16 to our FP8 format.

Run on the RTX Pro 6000 (96 GB VRAM) after downloading the BF16 model:
    python3 quantize_fp8.py
    python3 quantize_fp8.py /workspace/models/wan22-i2v-bf16 /workspace/models/wan22-i2v-qfp8

What it does
------------
1. Loads both DiTs as BF16 — both fit in 96 GB VRAM simultaneously.
2. For every nn.Linear weight:
       scale  = max(|W|) / 448        (448 = max of float8_e4m3fn)
       fp8_W  = (W / scale).to(float8_e4m3fn)
3. Saves per-block safetensors with ABSOLUTE state-dict keys:
       blocks.0.self_attn.q.weight        → float8_e4m3fn
       blocks.0.self_attn.q.weight.scale  → bfloat16 scalar
       blocks.0.self_attn.q.bias          → bfloat16  (not quantized)
       blocks.0.norm1.weight              → bfloat16  (not quantized)
4. Non-linear params (norms, Conv3d, embeddings) stay BF16.
5. T5, VAE, tokenizer copied unchanged from the BF16 source.
6. Writes format.json so model_server.py picks the FP8 loader automatically.

Output
------
{OUTPUT_DIR}/
  high_noise_model/   blocks.0.safetensors … blocks.39.safetensors, head.safetensors, …
  low_noise_model/    (same structure)
  text_encoder/       copied from BF16
  tokenizer/          copied from BF16
  vae/                copied from BF16
  format.json
"""
import gc
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BF16_DIR   = sys.argv[1] if len(sys.argv) > 1 else "/data/wan22-bf16"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "/workspace/models/wan22-i2v-qfp8"
WAN22_SRC  = sys.argv[3] if len(sys.argv) > 3 else "/workspace/wan22"
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn

sys.path.insert(0, WAN22_SRC)


def _top_group(key: str) -> str:
    """Map an absolute state-dict key to the shard file stem."""
    parts = key.split('.')
    if parts[0] == 'blocks':
        return f"blocks.{parts[1]}"
    return parts[0]


def _is_linear_weight(key: str, tensor: torch.Tensor) -> bool:
    """True if this is a quantisable nn.Linear weight (2-D, not norm/embedding)."""
    if tensor.dim() != 2 or not key.endswith('.weight'):
        return False
    lower = key.lower()
    return not any(tok in lower for tok in ('norm', 'embed', 'patch_embed', 'pos_embed'))


def _quantize(w: torch.Tensor):
    """Per-tensor absmax → float8_e4m3fn.  Returns (fp8_weight, bf16_scale)."""
    w_f32 = w.float()
    scale  = (w_f32.abs().max() / 448.0).clamp(min=1e-12)
    fp8_w  = (w_f32 / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return fp8_w, scale.to(torch.bfloat16)


def _save_dit(model: nn.Module, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()

    groups: dict[str, dict] = defaultdict(dict)
    n_fp8 = 0

    for key, tensor in state.items():
        group = _top_group(key)
        if _is_linear_weight(key, tensor):
            fp8_w, scale = _quantize(tensor)
            groups[group][key]            = fp8_w
            groups[group][f"{key}.scale"] = scale
            n_fp8 += 1
        else:
            groups[group][key] = tensor.to(torch.bfloat16)

    print(f"    {len(state)} params: {n_fp8} linear → FP8, {len(state)-n_fp8} other → BF16")

    from safetensors.torch import save_file
    for name in sorted(groups):
        save_file(groups[name], str(out_dir / f"{name}.safetensors"))

    print(f"    Saved {len(groups)} shards → {out_dir}")


def main():
    bf16_dir   = Path(BF16_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not bf16_dir.exists():
        sys.exit(f"BF16 model not found: {bf16_dir}\nRun download.py first.")

    from wan import WanI2V
    from wan.configs import WAN_CONFIGS

    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print(f"VRAM:  {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")
    print(f"BF16:  {bf16_dir}")
    print(f"Out:   {output_dir}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    import logging
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    cfg = WAN_CONFIGS['i2v-A14B']

    print("Loading BF16 model ...")
    pipe = WanI2V(
        config=cfg,
        checkpoint_dir=str(bf16_dir),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=False,       # 96 GB VRAM: load both DiTs to GPU
        convert_model_dtype=False,
    )

    for attr, subdir in [
        ('high_noise_model', 'high_noise_model'),
        ('low_noise_model',  'low_noise_model'),
    ]:
        model = getattr(pipe, attr).cpu()
        print(f"\nQuantizing {attr} ...")
        _save_dit(model, output_dir / subdir)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Copy T5, VAE, tokenizer unchanged
    print("\nCopying T5 / VAE / tokenizer ...")
    for name in ('text_encoder', 'tokenizer', 'vae', 'scheduler'):
        src = bf16_dir / name
        if src.exists():
            dst = output_dir / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  {name}/")

    for name in ('model_index.json', 'config.json'):
        src = bf16_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            print(f"  {name}")

    # Format marker — model_server.py reads this to select the FP8 loader
    fmt = {
        "format":       "wan22-i2v-qfp8-v1",
        "quant":        "float8_e4m3fn",
        "scale":        "per_tensor_absmax",
        "non_linear":   "bfloat16",
        "high_noise_dir": "high_noise_model",
        "low_noise_dir":  "low_noise_model",
    }
    (output_dir / 'format.json').write_text(json.dumps(fmt, indent=2))

    print(f"\n✓ Done → {output_dir}")
    print("\nNext: copy the output to your production network volume")
    print("      and set MODEL_PATH to point at it.")
    print("      model_server.py will detect format.json and use the FP8 loader.")


if __name__ == '__main__':
    main()
