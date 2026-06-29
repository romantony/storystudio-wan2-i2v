#!/usr/bin/env python
"""
Persistent Model Server for Wan2.2-I2V-A14B-FP8
Diffusers-based pipeline optimised for RTX 5090 (32 GB VRAM).

Runs as a background process. Loads model once on startup, then accepts
generation jobs via Unix socket — eliminating repeated load time between jobs.
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

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/wan22-i2v-fp8")
SOCKET_PATH = "/tmp/wan2_model_server.sock"

# height, width, flow_shift per resolution
RESOLUTIONS = {
    "480p": (480, 832, 3.0),
    "720p": (720, 1280, 5.0),
}


class ModelServer:
    def __init__(self):
        self.pipe = None

    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline

        print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram_gb:.1f} GB")

        if not Path(MODEL_PATH).exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. "
                "Download model weights to the network volume before starting."
            )

        print(f"Loading FP8 model from {MODEL_PATH} ...")
        start = time.time()

        # FP8 weights are stored on disk; dequantised to BF16 at compute time.
        # enable_model_cpu_offload() moves the T5 encoder back to CPU after the
        # initial text encoding step, keeping peak VRAM under 32 GB on RTX 5090.
        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.pipe.enable_model_cpu_offload()

        elapsed = time.time() - start
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"✓ Model loaded in {elapsed:.1f}s — {allocated:.1f}GB allocated / {reserved:.1f}GB reserved")

    def generate_video(self, params: dict) -> dict:
        import torch
        from diffusers.utils import export_to_video
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

            height, width, flow_shift = RESOLUTIONS[resolution]

            image = Image.open(image_path).convert("RGB")
            torch.cuda.empty_cache()

            print(f"Generating {resolution} ({width}x{height}), {frame_num} frames, {sample_steps} steps")

            result = self.pipe(
                image=image,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=frame_num,
                num_inference_steps=sample_steps,
                guidance_scale=5.0,
                flow_shift=flow_shift,
            )
            frames = result.frames[0]

            torch.cuda.empty_cache()
            export_to_video(frames, output_path, fps=16)

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
