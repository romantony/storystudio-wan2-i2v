"""
RunPod Serverless Handler for Wan2.2 I2V (Image-to-Video)
Version 2.0 - With Persistent Model Server for warm model between jobs

Key improvement: Model loads ONCE on container startup and stays warm,
eliminating the ~6 minute model load time for subsequent jobs.
"""
# FIRST: Set CUDA environment variables before ANY imports
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

# Now import other modules
import runpod
import subprocess
import tempfile
import base64
import time
import socket
import json
import boto3
import requests
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from huggingface_hub import snapshot_download

# Configuration
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models")
WAN_DIR = "/workspace/Wan2.2"
SOCKET_PATH = "/tmp/wan2_model_server.sock"
MODEL_SERVER_SCRIPT = "/workspace/handler/model_server.py"

# R2 Configuration
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "620baa808df08b1a30d448989365f7dd")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "a69ca34cdcdeb60bad5ed1a07a0dd29d")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "751a95202a9fa1eb9ff7d45e0bba5b57b0c2d1f0d45129f5f67c2486d5d4ae24")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "storystudio")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "parentearn.com")
R2_FOLDER = "VideoGen"

# Resolution mapping
RESOLUTION_MAP = {
    "480p": "832*480",
    "720p": "1280*720"
}

# Global state
model_server_process = None

def ensure_model_downloaded():
    """Download model if not cached"""
    model_dir = f"{MODEL_CACHE_DIR}/{MODEL_ID}"
    required_file = Path(model_dir) / "models_t5_umt5-xxl-enc-bf16.pth"
    
    if not required_file.exists():
        print(f"Downloading model (~11GB) to {model_dir}...")
        os.makedirs(model_dir, exist_ok=True)
        
        os.environ["HF_HOME"] = model_dir
        os.environ["HF_HUB_CACHE"] = model_dir
        
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        print("✓ Model downloaded")
    else:
        print("✓ Model found in cache")

def start_model_server():
    """Start the persistent model server as a background process"""
    global model_server_process
    
    if model_server_process is not None:
        print("Model server already running")
        return
    
    print("Starting persistent model server...")
    
    # Start model server in background
    model_server_process = subprocess.Popen(
        ["python", "-u", MODEL_SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait for server to be ready (model to be loaded)
    print("Waiting for model to load into GPU memory...")
    start_time = time.time()
    max_wait = 600  # 10 minutes max
    
    while time.time() - start_time < max_wait:
        if os.path.exists(SOCKET_PATH):
            # Try to connect
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(SOCKET_PATH)
                sock.close()
                print(f"✓ Model server ready! (took {time.time() - start_time:.1f}s)")
                return
            except:
                pass
        
        # Read and print server output
        if model_server_process.stdout:
            line = model_server_process.stdout.readline()
            if line:
                print(f"[ModelServer] {line.rstrip()}")
        
        time.sleep(1)
    
    raise Exception("Model server failed to start within timeout")

def send_to_model_server(request: dict, timeout: int = 3600) -> dict:
    """Send a job request to the model server with robust error handling"""
    global model_server_process
    
    # Check if model server is still running
    if model_server_process is not None:
        poll = model_server_process.poll()
        if poll is not None:
            print(f"WARNING: Model server process died with code {poll}")
            # Try to get any remaining output
            if model_server_process.stdout:
                remaining = model_server_process.stdout.read()
                if remaining:
                    print(f"Model server final output: {remaining}")
            raise Exception(f"Model server process died unexpectedly (exit code: {poll})")
    
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    
    try:
        sock.connect(SOCKET_PATH)
    except socket.error as e:
        raise Exception(f"Failed to connect to model server: {e}. Server may have crashed.")
    
    # Send request
    message = json.dumps(request) + "\n\n"
    sock.sendall(message.encode())
    
    # Receive response with better handling
    data = b""
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                # Connection closed
                break
            data += chunk
            if b"\n" in data:
                break
    except socket.timeout:
        sock.close()
        raise Exception(f"Socket read timeout after {timeout}s - generation may still be running")
    
    sock.close()
    
    # Check for empty response
    response_str = data.decode().strip()
    if not response_str:
        # Check if server is still alive
        if model_server_process is not None:
            poll = model_server_process.poll()
            if poll is not None:
                raise Exception(f"Model server crashed during generation (exit code: {poll})")
        raise Exception("Empty response from model server - server may have crashed or timed out")
    
    try:
        return json.loads(response_str)
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON from model server: {e}. Response was: {response_str[:500]}")

def get_r2_client():
    """Initialize and return R2 S3 client"""
    return boto3.client(
        's3',
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto'
    )

def upload_to_r2(file_path: str, filename: str) -> str:
    """Upload file to R2 and return public URL"""
    s3_client = get_r2_client()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    r2_key = f"{R2_FOLDER}/{timestamp}_{filename}"
    
    with open(file_path, 'rb') as f:
        s3_client.upload_fileobj(f, R2_BUCKET_NAME, r2_key)
    
    public_url = f"https://{R2_PUBLIC_URL}/{r2_key}"
    return public_url

def generate_video(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function for Image-to-Video
    Now uses persistent model server for warm model between jobs.
    """
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        # Validate inputs
        if "image" not in job_input:
            return {"error": "Missing required input: image"}
        
        # Get parameters
        image_input = job_input["image"]
        prompt = job_input.get("prompt", "")
        resolution = job_input.get("resolution", "480p")
        sample_steps = job_input.get("sample_steps", 30)
        frame_num = job_input.get("frame_num", 81)
        
        # Validate parameters
        if resolution not in RESOLUTION_MAP:
            return {"error": f"Invalid resolution. Must be one of: {list(RESOLUTION_MAP.keys())}"}
        
        if not (15 <= sample_steps <= 50):
            return {"error": "sample_steps must be between 15 and 50"}
        
        if not (17 <= frame_num <= 161):
            return {"error": "frame_num must be between 17 and 161"}
        
        # Handle image input (URL or base64)
        if image_input.startswith(('http://', 'https://')):
            print(f"Downloading image from URL: {image_input}")
            response = requests.get(image_input, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
        else:
            image_bytes = base64.b64decode(image_input)
        
        # Create temp directory for this job
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded image
            image_path = Path(tmpdir) / "input_image.jpg"
            image_path.write_bytes(image_bytes)
            
            # Output path for video
            output_path = Path(tmpdir) / "output_video.mp4"
            
            # Send request to model server
            print(f"Sending job to warm model server...")
            request = {
                "job_id": job_id,
                "image_path": str(image_path),
                "prompt": prompt,
                "resolution": resolution,
                "sample_steps": sample_steps,
                "frame_num": frame_num,
                "output_path": str(output_path)
            }
            
            result = send_to_model_server(request)
            
            if not result.get("success"):
                return {"error": result.get("error", "Generation failed")}
            
            # Get video size
            video_size_mb = round(output_path.stat().st_size / 1024 / 1024, 2)
            
            # Upload to R2
            print(f"Uploading video to R2...")
            video_filename = f"i2v_{resolution}_{sample_steps}steps.mp4"
            video_url = upload_to_r2(str(output_path), video_filename)
            
            generation_time = time.time() - start_time
            
            return {
                "video_url": video_url,
                "generation_time": round(generation_time, 2),
                "model_generation_time": round(result.get("generation_time", 0), 2),
                "video_size_mb": video_size_mb,
                "resolution": resolution,
                "sample_steps": sample_steps,
                "frame_num": frame_num
            }
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Initialize on container startup
print("=" * 60)
print("Wan2.2 I2V Handler v2.0 - Warm Model Architecture")
print("=" * 60)

# Apply FlashAttention patches
print("Applying FlashAttention compatibility patches...")
from patches.apply_patches import apply_flashattention_patches, apply_i2v_only_patches
apply_flashattention_patches()
apply_i2v_only_patches()
print("✓ Patches applied")

# Ensure model is downloaded
ensure_model_downloaded()

# Start persistent model server (loads model into GPU)
start_model_server()

print("✓ Handler ready with warm model!")
print("=" * 60)

# Start RunPod serverless handler
runpod.serverless.start({"handler": generate_video})
