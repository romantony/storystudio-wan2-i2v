#!/usr/bin/env python
"""
Persistent Model Server for Wan2.2 I2V
Keeps the model warm in GPU memory between jobs to eliminate 6-minute load time.

This server runs as a background process and receives job requests via a Unix socket.
The model is loaded once on startup and kept in memory for all subsequent jobs.
"""
import os
import sys
import json
import socket
import tempfile
import time
import traceback
from pathlib import Path

# Set CUDA environment before any imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Configuration
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models")
WAN_DIR = "/workspace/Wan2.2"
SOCKET_PATH = "/tmp/wan2_model_server.sock"

# Resolution mapping
RESOLUTION_MAP = {
    "480p": "832*480",
    "720p": "1280*720"
}

class ModelServer:
    def __init__(self):
        self.model_dir = f"{MODEL_CACHE_DIR}/{MODEL_ID}"
        self.pipe = None
        self.image_encoder = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the model into GPU memory once"""
        if self.model_loaded:
            print("Model already loaded, skipping...")
            return
            
        print("=" * 60)
        print("LOADING MODEL INTO GPU MEMORY (ONE-TIME)")
        print("=" * 60)
        
        start_time = time.time()
        
        # Initialize PyTorch and CUDA
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            # Warm up CUDA
            _ = torch.zeros(1, device="cuda")
            print("CUDA initialized")
        
        # Patch safetensors for CPU loading (workaround)
        from safetensors import torch as safetensors_torch
        _original_load = safetensors_torch.load_file
        def _patched_load(filename, device="cpu"):
            return _original_load(filename, device="cpu")
        safetensors_torch.load_file = _patched_load
        print("Patched safetensors for CPU loading")
        
        # Change to Wan2.2 directory
        os.chdir(WAN_DIR)
        sys.path.insert(0, WAN_DIR)
        
        # Import Wan2.2 modules
        from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
        from wan.image2video import WanI2V
        
        # Load model configuration
        cfg = WAN_CONFIGS["i2v-A14B"]
        
        print(f"Loading model from: {self.model_dir}")
        
        # Initialize the I2V pipeline
        self.pipe = WanI2V(
            config=cfg,
            checkpoint_dir=self.model_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            t5_cpu=False,
        )
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.1f} seconds")
        print("=" * 60)
        
        self.model_loaded = True
        
        # Print GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        
    def generate_video(self, params: dict) -> dict:
        """Generate a video using the warm model"""
        import torch
        from PIL import Image
        
        start_time = time.time()
        
        try:
            image_path = params["image_path"]
            prompt = params.get("prompt", "")
            resolution = params.get("resolution", "480p")
            sample_steps = params.get("sample_steps", 30)
            frame_num = params.get("frame_num", 81)
            output_path = params["output_path"]
            
            # Calculate max_area based on resolution
            # 480p = 832x480, 720p = 1280x720
            if resolution == "720p":
                max_area = 1280 * 720
                shift = 5.0
            else:  # 480p
                max_area = 832 * 480
                shift = 3.0  # Recommended for 480p
            
            print(f"Generating video: {resolution}, {sample_steps} steps, {frame_num} frames")
            print(f"Image: {image_path}")
            print(f"Prompt: {prompt[:100]}...")
            
            # Load image as PIL Image
            img = Image.open(image_path).convert("RGB")
            
            # Generate video using correct API
            video = self.pipe.generate(
                input_prompt=prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sampling_steps=sample_steps,
                seed=int(time.time()) % 2**32,
                offload_model=False,  # Keep model in GPU for speed
            )
            
            # Save video
            from wan.utils.utils import cache_video
            cache_video(
                tensor=video[None],
                save_file=output_path,
                fps=16,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            
            gen_time = time.time() - start_time
            print(f"✓ Video generated in {gen_time:.1f} seconds")
            
            return {
                "success": True,
                "output_path": output_path,
                "generation_time": gen_time
            }
            
        except Exception as e:
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def run(self):
        """Run the model server, listening for requests"""
        # Load model on startup
        self.load_model()
        
        # Remove old socket if exists
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        
        # Create Unix socket server
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(1)
        
        # Make socket accessible
        os.chmod(SOCKET_PATH, 0o777)
        
        print(f"Model server listening on {SOCKET_PATH}")
        print("Ready to receive jobs (model is warm!)")
        
        while True:
            try:
                conn, _ = server.accept()
                
                # Receive request
                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n\n" in data:
                        break
                
                # Parse request
                request = json.loads(data.decode().strip())
                print(f"\nReceived job request: {request.get('job_id', 'unknown')}")
                
                # Generate video
                result = self.generate_video(request)
                
                # Send response
                response = json.dumps(result) + "\n"
                conn.sendall(response.encode())
                conn.close()
                
            except Exception as e:
                print(f"Server error: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    server = ModelServer()
    server.run()
