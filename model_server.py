#!/usr/bin/env python
"""
Persistent Model Server for Wan2.2-I2V-A14B.

Supports two on-disk formats detected automatically via format.json:

  OUR FP8 FORMAT (wan22-i2v-qfp8-v1) — preferred, produced by quantize_wan22_fp8.py
  ─────────────────────────────────────────────────────────────────────────
  • nn.Linear weights stored as float8_e4m3fn + per-tensor BF16 scale.
  • Non-linear params (norms, embeddings, conv) kept as BF16.
  • Per-block safetensors with ABSOLUTE keys.
  • FP8Linear replaces nn.Linear at load time: weight stays FP8 on GPU,
    dequantized to BF16 on-the-fly for each matmul (temporary, freed after).
  • VRAM: both DiTs on GPU simultaneously (~14 GB FP8 each = ~28 GB total).
    No CPU↔GPU offload needed during inference.

  NALEXAND FORMAT (fallback, BF16 path)
  ─────────────────────────────────────
  • Per-block safetensors with relative or absolute keys.  Empty weight_map.
  • Weights streamed one shard at a time, cast FP8 → BF16, assigned directly.
  • high_noise moved to GPU after loading to free CPU RAM for low_noise.
  • VRAM: one DiT (~28.6 GB BF16) at a time; offload_model=True for inference.
"""
import gc
import os
import sys
import json
import socket
import time
import traceback
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

WAN22_PATH = "/workspace/wan22"
if WAN22_PATH not in sys.path:
    sys.path.insert(0, WAN22_PATH)

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/wan22-i2v-fp8")
SOCKET_PATH = "/tmp/wan2_model_server.sock"

RESOLUTIONS = {
    "480p": (832 * 480, 3.0),
    "720p": (1280 * 720, 5.0),
}


# ════════════════════════════════════════════════════════════════════════════
# FP8Linear — our own FP8 inference wrapper
# ════════════════════════════════════════════════════════════════════════════

class FP8Linear:
    """
    Drop-in nn.Linear replacement that stores weight in float8_e4m3fn.

    _fp8_w, _scale, _bias are plain Python attributes (not registered buffers/
    parameters) so that Module.to(dtype=bf16) never silently casts the FP8
    weight.  Device movement IS supported via the _apply() override, which
    propagates device changes while preserving the FP8 dtype.

    Forward: dequantize weight to input dtype, run F.linear, discard BF16 copy.
    Memory:  FP8 weight lives on GPU, BF16 copy is ~single-layer transient.
    """

    def __init__(self, fp8_w: "torch.Tensor", scale: "torch.Tensor",
                 bias: Optional["torch.Tensor"] = None):
        # Delay nn.Module import — model_server may be imported before torch
        import torch.nn as nn
        # We subclass nn.Module inline to get module registration
        if not isinstance(self, nn.Module):
            raise TypeError("FP8Linear must be used as a mixin with nn.Module")
        self._fp8_w = fp8_w
        self._scale  = scale
        self._bias   = bias

    def _apply(self, fn, recurse=True):
        """Called by Module.to() / .cuda() / .cpu() — move tensors to new device."""
        self._scale = fn(self._scale)
        if self._bias is not None:
            self._bias = fn(self._bias)
        # Apply fn to FP8 weight but preserve its dtype.
        result = fn(self._fp8_w)
        if result.dtype == self._fp8_w.dtype:
            self._fp8_w = result                        # device-only change: safe
        else:
            self._fp8_w = self._fp8_w.to(result.device) # fn tried to cast dtype: keep FP8, just move device
        return self

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        import torch.nn.functional as F
        w = self._fp8_w.to(x.dtype) * self._scale      # BF16 weight, freed after matmul
        return F.linear(x, w, self._bias)

    def extra_repr(self) -> str:
        return (f"in={self._fp8_w.shape[1]}, out={self._fp8_w.shape[0]}, "
                f"bias={self._bias is not None}, weight=fp8_e4m3fn")


def _make_fp8_linear_class() -> type:
    """Return a class that is both nn.Module and FP8Linear (avoids top-level import)."""
    import torch.nn as nn

    class _FP8Linear(FP8Linear, nn.Module):
        def __init__(self, fp8_w, scale, bias=None):
            nn.Module.__init__(self)
            FP8Linear.__init__(self, fp8_w, scale, bias)

        def _apply(self, fn, recurse=True):
            return FP8Linear._apply(self, fn, recurse)

        def forward(self, x):
            return FP8Linear.forward(self, x)

        def extra_repr(self):
            return FP8Linear.extra_repr(self)

    return _FP8Linear


_FP8LinearClass = None   # created lazily after torch is imported


def _fp8_linear_cls():
    global _FP8LinearClass
    if _FP8LinearClass is None:
        _FP8LinearClass = _make_fp8_linear_class()
    return _FP8LinearClass


# ════════════════════════════════════════════════════════════════════════════
# Shared parameter-navigation helper
# ════════════════════════════════════════════════════════════════════════════

def _get_module(model: "torch.nn.Module", key: str) -> "torch.nn.Module":
    """Navigate to the sub-module named by key (dot-separated path)."""
    m = model
    for part in key.split('.'):
        m = getattr(m, part)
    return m


def _assign_param(model: "torch.nn.Module", key: str, value: "torch.Tensor") -> None:
    """Replace a model parameter at state-dict key with value tensor."""
    import torch
    parts = key.split('.')
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], torch.nn.Parameter(value, requires_grad=False))


# ════════════════════════════════════════════════════════════════════════════
# OUR FP8 FORMAT LOADER
# ════════════════════════════════════════════════════════════════════════════

def _load_our_fp8_model(model, block_dir: str) -> None:
    """
    Load our quantize_wan22_fp8.py output into a WanModel.

    For each shard file:
      - Tensors with dtype float8_e4m3fn paired with a companion .weight.scale
        → the parent nn.Linear is replaced with an FP8Linear (stays FP8 on GPU).
      - All other tensors → assigned as BF16 parameters.

    Processing is shard-by-shard so peak extra RAM during loading is ~one shard
    (~0.3 GB FP8) regardless of the total model size.
    """
    import torch
    from safetensors.torch import load_file
    Cls = _fp8_linear_cls()

    block_dir_path = Path(block_dir)
    shard_files = sorted(block_dir_path.glob("*.safetensors"))
    if not shard_files:
        raise RuntimeError(f"No safetensors files found in {block_dir}")

    total_loaded = 0
    total_fp8    = 0
    total_bf16   = 0

    for shard_file in shard_files:
        tensors = load_file(str(shard_file))

        # Identify which base keys have a companion .scale (these are FP8 weights)
        fp8_weight_keys = {
            k for k in tensors
            if not k.endswith('.scale') and f"{k}.scale" in tensors
        }
        # Bias keys that belong to FP8 linear layers (handled alongside weight)
        fp8_bias_keys = {
            k for k in tensors
            if not k.endswith('.scale')
            and k.endswith('.bias')
            and k[:-len('.bias')] + '.weight' in fp8_weight_keys
        }

        for key, tensor in tensors.items():
            if key.endswith('.scale'):
                continue                    # processed below alongside the weight

            if key in fp8_weight_keys:
                # --- FP8 linear weight: replace the nn.Linear with FP8Linear ---
                scale     = tensors[f"{key}.scale"]
                bias_key  = key[:-len('.weight')] + '.bias'
                bias      = tensors.get(bias_key, None)
                if bias is not None:
                    bias = bias.to(torch.bfloat16)

                fp8_linear = Cls(tensor, scale, bias)   # tensor stays float8_e4m3fn

                # Navigate to parent module and replace
                parent_path, leaf = key.rsplit('.', 1)  # leaf == 'weight'
                module_path = parent_path               # e.g. 'blocks.0.self_attn.q'
                grandparent_path, module_name = (
                    module_path.rsplit('.', 1) if '.' in module_path
                    else ('', module_path)
                )
                if grandparent_path:
                    grandparent = _get_module(model, grandparent_path)
                else:
                    grandparent = model
                setattr(grandparent, module_name, fp8_linear)
                total_fp8  += 1
                total_loaded += 1

            elif key in fp8_bias_keys:
                continue                    # already handled in the weight block above

            else:
                # --- BF16 parameter ---
                _assign_param(model, key, tensor.to(torch.bfloat16))
                total_bf16   += 1
                total_loaded += 1

        del tensors
        gc.collect()

    print(f"    {Path(block_dir).name}: "
          f"{total_loaded} loaded ({total_fp8} FP8 linears, {total_bf16} BF16 params)")


# ════════════════════════════════════════════════════════════════════════════
# NALEXAND FORMAT LOADER  (BF16 fallback)
# ════════════════════════════════════════════════════════════════════════════

def _load_perblock_weights(model, block_dir: str) -> None:
    """
    Stream-load nalexand per-block safetensors → BF16.
    One shard at a time to keep peak CPU RAM ~1 GB per shard.
    Keys may be relative (self_attn.q.weight) or absolute (blocks.0.self_attn.q.weight).
    """
    import torch
    from safetensors.torch import load_file

    block_dir_path = Path(block_dir)
    shard_files = sorted(block_dir_path.glob("*.safetensors"))
    if not shard_files:
        raise RuntimeError(f"No safetensors files found in {block_dir}")

    model_keys    = set(model.state_dict().keys())
    loaded        = 0
    missing_keys  = set(model_keys)
    unexpected_keys = []

    for shard_file in shard_files:
        module_prefix = shard_file.stem
        tensors = load_file(str(shard_file))

        for raw_key, tensor in tensors.items():
            candidate_abs = f"{module_prefix}.{raw_key}"
            if candidate_abs in model_keys:
                abs_key = candidate_abs
            elif raw_key in model_keys:
                abs_key = raw_key
            else:
                unexpected_keys.append(candidate_abs)
                continue

            _assign_param(model, abs_key, tensor.to(torch.bfloat16))
            missing_keys.discard(abs_key)
            loaded += 1

        del tensors
        gc.collect()

    missing = len(missing_keys)
    extra   = len(unexpected_keys)
    print(f"    {Path(block_dir).name}: {loaded} loaded, {missing} missing, {extra} unexpected")
    if missing_keys:
        print(f"    First 5 missing: {list(missing_keys)[:5]}")
    if unexpected_keys:
        print(f"    First 5 unexpected: {unexpected_keys[:5]}")


# ════════════════════════════════════════════════════════════════════════════
# Model server
# ════════════════════════════════════════════════════════════════════════════

class ModelServer:
    def __init__(self):
        self.pipe = None
        self.using_fp8_format = False

    def load_model(self):
        import torch
        from wan import WanI2V
        from wan.configs import WAN_CONFIGS

        print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram_gb:.1f} GB")

        if not Path(MODEL_PATH).exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}.")

        # Detect format
        fmt_file = Path(MODEL_PATH) / 'format.json'
        if fmt_file.exists():
            fmt = json.loads(fmt_file.read_text())
            if fmt.get('format') == 'wan22-i2v-qfp8-v1':
                self.using_fp8_format = True
                high_noise_dir = fmt.get('high_noise_dir', 'high_noise_model')
                low_noise_dir  = fmt.get('low_noise_dir',  'low_noise_model')
                print("Format: our FP8 (float8_e4m3fn + per-tensor scale) — both DiTs on GPU")
            else:
                raise RuntimeError(f"Unknown format: {fmt.get('format')}")
        else:
            self.using_fp8_format = False
            high_noise_dir = 'high_noise_model_fp8'
            low_noise_dir  = 'low_noise_model_fp8'
            print("Format: nalexand per-block FP8 → BF16 (fallback path)")

        print(f"Loading model from {MODEL_PATH} ...")
        start = time.time()

        cfg = WAN_CONFIGS['i2v-A14B']
        cfg.high_noise_checkpoint = high_noise_dir
        cfg.low_noise_checkpoint  = low_noise_dir

        import logging
        logging.getLogger("diffusers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        self.pipe = WanI2V(
            config=cfg,
            checkpoint_dir=MODEL_PATH,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
            init_on_cpu=True,           # DiTs start on meta device
            convert_model_dtype=False,
        )

        if self.using_fp8_format:
            self._load_our_fp8(high_noise_dir, low_noise_dir)
        else:
            self._load_nalexand_fp8(high_noise_dir, low_noise_dir)

        elapsed   = time.time() - start
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        print(f"✓ Model ready in {elapsed:.1f}s — {allocated:.1f} GB alloc / {reserved:.1f} GB reserved")

    def _load_our_fp8(self, high_noise_dir: str, low_noise_dir: str) -> None:
        """FP8 path: load both DiTs directly onto GPU as FP8.  No offload needed."""
        import torch

        print("Loading DiT weights (our FP8 format) → FP8 on GPU ...")
        for attr, subdir in [
            ('high_noise_model', high_noise_dir),
            ('low_noise_model',  low_noise_dir),
        ]:
            model     = getattr(self.pipe, attr)
            block_dir = os.path.join(MODEL_PATH, subdir)
            _load_our_fp8_model(model, block_dir)
            model.to(torch.device('cuda', 0))   # move FP8 + BF16 buffers to GPU

        vram_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"    Both DiTs on GPU: {vram_gb:.1f} GB VRAM used")

    def _load_nalexand_fp8(self, high_noise_dir: str, low_noise_dir: str) -> None:
        """Nalexand fallback: stream FP8 → BF16.  Move high_noise to GPU to free CPU RAM."""
        import torch

        print("Loading DiT weights (nalexand per-block format) → BF16 ...")
        for attr, subdir in [
            ('high_noise_model', high_noise_dir),
            ('low_noise_model',  low_noise_dir),
        ]:
            model     = getattr(self.pipe, attr)
            block_dir = os.path.join(MODEL_PATH, subdir)
            _load_perblock_weights(model, block_dir)

            if attr == 'high_noise_model':
                print("    Moving high_noise_model to GPU ...")
                model.to(torch.device('cuda', 0))
                gc.collect()

    def generate_video(self, params: dict) -> dict:
        import torch
        import numpy as np
        import imageio
        from PIL import Image

        start = time.time()
        try:
            image_path   = params["image_path"]
            prompt       = params.get("prompt", "")
            resolution   = params.get("resolution", "480p")
            sample_steps = params.get("sample_steps", 25)
            frame_num    = params.get("frame_num", 81)
            output_path  = params["output_path"]

            if resolution not in RESOLUTIONS:
                raise ValueError(f"Unknown resolution '{resolution}'. Choose 480p or 720p.")

            max_area, flow_shift = RESOLUTIONS[resolution]
            image = Image.open(image_path).convert("RGB")
            torch.cuda.empty_cache()

            print(f"Generating {resolution} (max_area={max_area}), {frame_num} frames, {sample_steps} steps")

            # FP8 path: both DiTs already on GPU, no offload needed.
            # Nalexand path: offload_model=True swaps the inactive DiT to CPU.
            offload = not self.using_fp8_format

            video_tensor = self.pipe.generate(
                input_prompt=prompt,
                img=image,
                max_area=max_area,
                frame_num=frame_num,
                shift=flow_shift,
                sample_solver='unipc',
                sampling_steps=sample_steps,
                guide_scale=5.0,
                n_prompt="",
                seed=-1,
                offload_model=offload,
            )

            # (C, N, H, W) [-1,1] → (N, H, W, C) uint8
            frames = (video_tensor.clamp(-1, 1) + 1) / 2
            frames = frames.permute(1, 2, 3, 0).cpu().float().numpy()
            frames = (frames * 255).astype(np.uint8)

            torch.cuda.empty_cache()

            writer = imageio.get_writer(output_path, fps=16, codec='libx264', quality=8)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

            if not Path(output_path).exists():
                raise RuntimeError(f"Video not created at {output_path}")

            size_mb  = Path(output_path).stat().st_size / 1024 / 1024
            gen_time = time.time() - start
            print(f"✓ Done in {gen_time:.1f}s — {size_mb:.1f} MB")

            return {
                "success": True,
                "output_path": output_path,
                "generation_time": gen_time,
                "file_size_mb": round(size_mb, 2),
            }

        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def run(self):
        import signal

        def shutdown(signum, frame):
            print(f"Signal {signum} — shutting down")
            sys.exit(0)

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)

        self.load_model()

        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(1)
        os.chmod(SOCKET_PATH, 0o777)

        print(f"✓ Model server ready on {SOCKET_PATH}")

        while True:
            conn = None
            try:
                conn, _ = server.accept()
                conn.settimeout(5.0)

                data = b""
                try:
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                        if b"\n\n" in data:
                            break
                except socket.timeout:
                    pass

                request_str = data.decode().strip()
                if not request_str:
                    conn.close()
                    continue

                request = json.loads(request_str)
                print(f"\nJob {request.get('job_id', 'unknown')}")

                result = self.generate_video(request)
                conn.sendall((json.dumps(result) + "\n").encode())

            except Exception as e:
                print(f"Server error: {e}")
                traceback.print_exc()
                if conn:
                    try:
                        conn.sendall(
                            (json.dumps({"success": False, "error": str(e)}) + "\n").encode()
                        )
                    except Exception:
                        pass
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass


if __name__ == "__main__":
    ModelServer().run()
