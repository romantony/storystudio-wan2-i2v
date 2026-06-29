#!/usr/bin/env python
"""
Persistent Model Server for Wan2.2-I2V-A14B-FP8
Uses native Wan2.2 code with FP8 checkpoint directory override.
Optimised for RTX 5090 (32 GB VRAM, native FP8).

Both 14B DiT experts load in FP8 (14.3 GB each = 28.6 GB total).
T5 encoder (11.4 GB) stays on CPU. offload_model=True during generate
swaps the inactive DiT to CPU so peak GPU usage stays under 32 GB.
"""
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

# Wan2.2 native code cloned into the image at build time
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

        print(f"Loading FP8 model from {MODEL_PATH} ...")
        start = time.time()

        # Get A14B I2V config and redirect to FP8 checkpoint directories.
        # nalexand stores actual weights in high_noise_model_fp8/ and low_noise_model_fp8/;
        # the plain high_noise_model/ and low_noise_model/ dirs have empty weight maps.
        cfg = WAN_CONFIGS['i2v-A14B']
        cfg.high_noise_checkpoint = 'high_noise_model_fp8'
        cfg.low_noise_checkpoint = 'low_noise_model_fp8'

        self.pipe = WanI2V(
            config=cfg,
            checkpoint_dir=MODEL_PATH,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,         # keep 11.4 GB T5 on CPU
            init_on_cpu=True,
            convert_model_dtype=False,  # keep FP8 dtype; both DiTs = 28.6 GB
        )

        elapsed = time.time() - start
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"✓ Model loaded in {elapsed:.1f}s — {allocated:.1f} GB allocated / {reserved:.1f} GB reserved")

    def generate_video(self, params: dict) -> dict:
        import torch
        import numpy as np
        import imageio
        from PIL import Image

        start = time.time()
        try:
            image_path = params["image_path"]
            prompt = params.get("prompt", "")
            resolution = params.get("resolution", "480p")
            sample_steps = params.get("sample_steps", 25)
            frame_num = params.get("frame_num", 81)
            output_path = params["output_path"]

            if resolution not in RESOLUTIONS:
                raise ValueError(f"Unknown resolution '{resolution}'. Choose 480p or 720p.")

            max_area, flow_shift = RESOLUTIONS[resolution]

            image = Image.open(image_path).convert("RGB")
            torch.cuda.empty_cache()

            print(f"Generating {resolution} (max_area={max_area}), {frame_num} frames, {sample_steps} steps")

            # generate() returns tensor (C, N, H, W) in range [-1, 1]
            # offload_model=True swaps the inactive DiT to CPU during sampling
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

            # Convert (C, N, H, W) [-1, 1] tensor → (N, H, W, C) uint8 numpy
            frames = (video_tensor.clamp(-1, 1) + 1) / 2       # [0, 1]
            frames = frames.permute(1, 2, 3, 0).cpu().float().numpy()  # (N, H, W, C)
            frames = (frames * 255).astype(np.uint8)

            torch.cuda.empty_cache()

            writer = imageio.get_writer(output_path, fps=16, codec='libx264', quality=8)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

            if not Path(output_path).exists():
                raise RuntimeError(f"Video file was not created at {output_path}")

            size_mb = Path(output_path).stat().st_size / 1024 / 1024
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
            print(f"Signal {signum} received — shutting down")
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
                        conn.sendall((json.dumps({"success": False, "error": str(e)}) + "\n").encode())
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
