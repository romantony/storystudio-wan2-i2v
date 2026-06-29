#!/usr/bin/env python
"""
Persistent Model Server for Wan2.2-I2V-A14B-FP8
Native Wan2.2 code with custom per-block FP8 weight loader.

The nalexand FP8 model stores DiT weights as one safetensors file per block
(blocks.0.safetensors … blocks.39.safetensors, head.safetensors, etc.) with
an empty weight_map in the index.json.  Standard WanModel.from_pretrained()
loads nothing from this format, resulting in randomly-initialised weights.

We let WanI2V load T5/VAE normally, then replace the DiT parameters by
scanning all *.safetensors files in the _fp8 subdirectories, processing one
shard (~300 MB FP8) at a time to avoid accumulating the full ~14 GB FP8
state dict in CPU RAM.  Each shard is loaded, cast to BF16, assigned directly
to the model parameter, then freed before the next shard is loaded.

After high_noise is fully loaded (BF16 on CPU), it is moved to GPU before
low_noise is loaded, keeping peak CPU RAM around 40 GB instead of ~70 GB.
With offload_model=True during generate(), only one 14B DiT (~28.6 GB BF16)
resides on the GPU at a time, fitting within 32 GB VRAM.
"""
import gc
import os
import sys
import json
import socket
import time
import traceback
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

WAN22_PATH = "/workspace/wan22"
if WAN22_PATH not in sys.path:
    sys.path.insert(0, WAN22_PATH)

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/wan22-i2v-fp8")
SOCKET_PATH = "/tmp/wan2_model_server.sock"

# Resolution → (max_area, flow_shift)
RESOLUTIONS = {
    "480p": (832 * 480, 3.0),
    "720p": (1280 * 720, 5.0),
}


def _assign_param(model: "torch.nn.Module", key: str, value: "torch.Tensor") -> None:
    """Replace a model parameter by navigating the module hierarchy via its state-dict key."""
    import torch
    parts = key.split('.')
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], torch.nn.Parameter(value, requires_grad=False))


def _load_perblock_weights(model, block_dir: str) -> None:
    """
    Stream-load nalexand per-block safetensors directly into model parameters.

    Each *.safetensors shard (~300 MB FP8) is loaded, cast to BF16, and assigned
    parameter-by-parameter to the model before the next shard is loaded.  This keeps
    peak extra CPU RAM per shard to ~1 GB instead of accumulating a full 14 GB FP8
    state dict — which would cause OOM when loading the second DiT on top of the first.

    Keys inside each shard file may be relative ('self_attn.q.weight') or absolute
    ('blocks.0.self_attn.q.weight').  We resolve by checking the model's own state_dict
    keys with the file stem as prefix, then fall back to the raw key directly.
    """
    import torch
    from safetensors.torch import load_file

    block_dir_path = Path(block_dir)
    shard_files = sorted(block_dir_path.glob("*.safetensors"))
    if not shard_files:
        raise RuntimeError(f"No safetensors files found in {block_dir}")

    model_keys = set(model.state_dict().keys())
    loaded = 0
    missing_keys = set(model_keys)
    unexpected_keys = []

    for shard_file in shard_files:
        module_prefix = shard_file.stem   # e.g. 'blocks.0', 'head', 'patch_embedding'
        tensors = load_file(str(shard_file))

        for raw_key, tensor in tensors.items():
            # Resolve key: try module_prefix.raw_key first (handles relative keys)
            candidate_abs = f"{module_prefix}.{raw_key}"
            if candidate_abs in model_keys:
                abs_key = candidate_abs
            elif raw_key in model_keys:
                abs_key = raw_key
            else:
                unexpected_keys.append(candidate_abs)
                continue

            # Cast FP8 → BF16, assign directly, then let the shard-level del free FP8
            bf16_tensor = tensor.to(torch.bfloat16)
            _assign_param(model, abs_key, bf16_tensor)
            missing_keys.discard(abs_key)
            loaded += 1

        # Free the entire FP8 shard before moving to the next one
        del tensors
        gc.collect()

    missing = len(missing_keys)
    extra = len(unexpected_keys)
    print(f"    {Path(block_dir).name}: {loaded} loaded, {missing} missing, {extra} unexpected")

    if missing_keys:
        print(f"    First 5 missing: {list(missing_keys)[:5]}")
    if unexpected_keys:
        print(f"    First 5 unexpected: {unexpected_keys[:5]}")


class ModelServer:
    def __init__(self):
        self.pipe = None

    def load_model(self):
        import torch
        from wan import WanI2V
        from wan.configs import WAN_CONFIGS

        print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram_gb:.1f} GB")

        if not Path(MODEL_PATH).exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. "
                "Download nalexand/Wan2.2-I2V-A14B-FP8 to the network volume first."
            )

        print(f"Loading model from {MODEL_PATH} ...")
        start = time.time()

        cfg = WAN_CONFIGS['i2v-A14B']
        cfg.high_noise_checkpoint = 'high_noise_model_fp8'
        cfg.low_noise_checkpoint  = 'low_noise_model_fp8'

        # Suppress verbose "Some weights not initialized" warnings from diffusers/transformers —
        # expected because the per-block FP8 index.json has an empty weight_map.
        import logging
        logging.getLogger("diffusers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        # WanI2V correctly loads T5 and VAE.
        # DiT models get architecture-only (meta device) because the per-block
        # safetensors format has an empty weight_map in the index.json.
        self.pipe = WanI2V(
            config=cfg,
            checkpoint_dir=MODEL_PATH,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,           # T5 (11.4 GB) stays on CPU
            init_on_cpu=True,
            convert_model_dtype=False,
        )

        # Stream-load FP8 weights one shard at a time into each DiT.
        # After high_noise is fully loaded to CPU as BF16, move it to GPU to free
        # CPU RAM before loading low_noise (avoids ~71 GB peak → ~40 GB peak).
        print("Loading FP8 DiT weights (per-block format) → BF16 ...")
        for attr, subdir in [
            ('high_noise_model', 'high_noise_model_fp8'),
            ('low_noise_model',  'low_noise_model_fp8'),
        ]:
            model = getattr(self.pipe, attr)
            block_dir = os.path.join(MODEL_PATH, subdir)
            _load_perblock_weights(model, block_dir)   # BF16 params on CPU

            if attr == 'high_noise_model':
                # Move to GPU now so CPU RAM is free for low_noise loading.
                # offload_model=True in generate() handles swapping during inference.
                print("    Moving high_noise_model to GPU ...")
                model.to(torch.device('cuda', 0))
                gc.collect()

        elapsed = time.time() - start
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        print(f"✓ Model ready in {elapsed:.1f}s — {allocated:.1f} GB alloc / {reserved:.1f} GB reserved")

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

            # generate() returns (C, N, H, W) tensor in range [-1, 1].
            # offload_model=True swaps the inactive DiT to CPU during sampling.
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
                offload_model=True,
            )

            # Convert (C, N, H, W) [-1,1] → (N, H, W, C) uint8
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
