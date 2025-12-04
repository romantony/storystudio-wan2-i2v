# Wan2.2 Image-to-Video API Documentation

## Overview

This API provides image-to-video generation using Wan2.2 I2V-A14B model hosted on RunPod serverless infrastructure.

**Base URL:** `https://api.runpod.ai/v2/zpms3tyvv2gelr`

**Authentication:** Bearer token in Authorization header

---

## API Endpoints

### 1. Generate Video (Async)

Start a video generation job asynchronously.

**Endpoint:** `POST /run`

**Headers:**
```
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
```

**Request Body:**
```json
{
  "input": {
    "image": "https://example.com/image.png",
    "prompt": "A slow push-in shot with camera movement...",
    "resolution": "480p",
    "sample_steps": 20,
    "frame_num": 81
  }
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | ✅ Yes | - | Image URL or base64-encoded image |
| `prompt` | string | No | "" | Text description of desired motion/animation |
| `resolution` | string | No | "480p" | Video resolution: "480p" or "720p" |
| `sample_steps` | int | No | 30 | Denoising steps (15-50). Lower = faster, higher = better quality |
| `frame_num` | int | No | 81 | Number of frames (17-161). 16 fps, so 81 = ~5 seconds |

**Response:**
```json
{
  "id": "job-id-here",
  "status": "IN_QUEUE"
}
```

---

### 2. Check Job Status

**Endpoint:** `GET /status/{job_id}`

**Response (In Progress):**
```json
{
  "id": "job-id-here",
  "status": "IN_PROGRESS"
}
```

**Response (Completed):**
```json
{
  "id": "job-id-here",
  "status": "COMPLETED",
  "output": {
    "video_url": "https://parentearn.com/VideoGen/20251204_155539_i2v_480p_20steps.mp4",
    "generation_time": 664.84,
    "video_size_mb": 4.03,
    "resolution": "480p",
    "sample_steps": 20,
    "frame_num": 81
  }
}
```

**Status Values:**
- `IN_QUEUE` - Job waiting for available worker
- `IN_PROGRESS` - Job is being processed
- `COMPLETED` - Job finished successfully
- `FAILED` - Job failed (check error field)

---

### 3. Generate Video (Sync)

Wait for completion in a single request (subject to timeout).

**Endpoint:** `POST /runsync`

**Note:** Not recommended for video generation due to long processing times.

---

### 4. Health Check

Check endpoint status and worker availability.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "jobs": {
    "completed": 10,
    "failed": 1,
    "inProgress": 1,
    "inQueue": 0,
    "retried": 0
  },
  "workers": {
    "idle": 0,
    "initializing": 1,
    "ready": 1,
    "running": 1,
    "throttled": 0
  }
}
```

---

### 5. Update Endpoint (Worker Management)

Enable or disable active workers dynamically.

**Endpoint:** `POST /update` (via management API)

**Full URL:** `https://api.runpod.ai/v2/{endpoint_id}/update`

**Request Body:**
```json
{
  "minWorkers": 1
}
```

---

## Worker Management Strategy

### Cost-Optimized Workflow

Enable workers only when processing a batch of videos, then disable to save costs.

```python
import requests
import time

class I2VClient:
    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    # ==========================================
    # WORKER MANAGEMENT
    # ==========================================
    
    def set_active_workers(self, count: int) -> dict:
        """
        Enable or disable active workers.
        
        Args:
            count: Number of minimum workers (0 = no active workers, 1+ = keep workers warm)
        
        Returns:
            API response
        """
        response = requests.post(
            f"{self.base_url}/update",
            headers=self.headers,
            json={"minWorkers": count}
        )
        return response.json()
    
    def enable_worker(self) -> dict:
        """Enable 1 active worker for faster processing."""
        print("🚀 Enabling active worker...")
        return self.set_active_workers(1)
    
    def disable_worker(self) -> dict:
        """Disable active workers to save costs."""
        print("💤 Disabling active worker...")
        return self.set_active_workers(0)
    
    def get_health(self) -> dict:
        """Get endpoint health and worker status."""
        response = requests.get(
            f"{self.base_url}/health",
            headers=self.headers
        )
        return response.json()
    
    def wait_for_worker_ready(self, timeout: int = 420, poll_interval: int = 10) -> bool:
        """
        Wait for at least one worker to be ready.
        
        Args:
            timeout: Maximum seconds to wait (default 7 minutes)
            poll_interval: Seconds between health checks
        
        Returns:
            True if worker is ready, False if timeout
        """
        print("⏳ Waiting for worker to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            health = self.get_health()
            workers = health.get("workers", {})
            ready = workers.get("ready", 0)
            running = workers.get("running", 0)
            initializing = workers.get("initializing", 0)
            
            print(f"   Workers - Ready: {ready}, Running: {running}, Initializing: {initializing}")
            
            if ready > 0 or running > 0:
                print("✅ Worker is ready!")
                return True
            
            time.sleep(poll_interval)
        
        print("❌ Timeout waiting for worker")
        return False
    
    # ==========================================
    # VIDEO GENERATION
    # ==========================================
    
    def generate_video(
        self,
        image: str,
        prompt: str = "",
        resolution: str = "480p",
        sample_steps: int = 20,
        frame_num: int = 81
    ) -> str:
        """
        Submit a video generation job.
        
        Args:
            image: Image URL or base64 string
            prompt: Motion/animation description
            resolution: "480p" or "720p"
            sample_steps: Denoising steps (15-50)
            frame_num: Number of frames (17-161)
        
        Returns:
            Job ID
        """
        response = requests.post(
            f"{self.base_url}/run",
            headers=self.headers,
            json={
                "input": {
                    "image": image,
                    "prompt": prompt,
                    "resolution": resolution,
                    "sample_steps": sample_steps,
                    "frame_num": frame_num
                }
            }
        )
        result = response.json()
        return result.get("id")
    
    def check_status(self, job_id: str) -> dict:
        """Check status of a job."""
        response = requests.get(
            f"{self.base_url}/status/{job_id}",
            headers=self.headers
        )
        return response.json()
    
    def wait_for_completion(self, job_id: str, poll_interval: int = 5) -> dict:
        """
        Wait for a job to complete.
        
        Args:
            job_id: The job ID to monitor
            poll_interval: Seconds between status checks
        
        Returns:
            Final job result
        """
        while True:
            status = self.check_status(job_id)
            job_status = status.get("status", "UNKNOWN")
            
            if job_status == "COMPLETED":
                return status.get("output", {})
            elif job_status == "FAILED":
                return {"error": status.get("error", "Job failed")}
            
            time.sleep(poll_interval)
    
    # ==========================================
    # BATCH PROCESSING
    # ==========================================
    
    def process_batch(self, jobs: list, auto_manage_workers: bool = True) -> list:
        """
        Process a batch of video generation jobs.
        
        Args:
            jobs: List of job configs, each with: image, prompt, resolution, sample_steps, frame_num
            auto_manage_workers: If True, enable worker before and disable after
        
        Returns:
            List of results
        """
        results = []
        
        try:
            # Enable worker if auto-managing
            if auto_manage_workers:
                self.enable_worker()
                self.wait_for_worker_ready()
            
            # Submit all jobs
            job_ids = []
            for i, job in enumerate(jobs):
                print(f"📤 Submitting job {i+1}/{len(jobs)}...")
                job_id = self.generate_video(
                    image=job["image"],
                    prompt=job.get("prompt", ""),
                    resolution=job.get("resolution", "480p"),
                    sample_steps=job.get("sample_steps", 20),
                    frame_num=job.get("frame_num", 81)
                )
                job_ids.append(job_id)
                print(f"   Job ID: {job_id}")
            
            # Wait for all jobs to complete
            for i, job_id in enumerate(job_ids):
                print(f"⏳ Waiting for job {i+1}/{len(jobs)}...")
                result = self.wait_for_completion(job_id)
                results.append(result)
                
                if "error" in result:
                    print(f"   ❌ Error: {result['error']}")
                else:
                    print(f"   ✅ Video: {result.get('video_url')}")
        
        finally:
            # Disable worker if auto-managing and no jobs in queue
            if auto_manage_workers:
                health = self.get_health()
                in_queue = health.get("jobs", {}).get("inQueue", 0)
                in_progress = health.get("jobs", {}).get("inProgress", 0)
                
                if in_queue == 0 and in_progress == 0:
                    self.disable_worker()
                else:
                    print(f"⚠️ Jobs still pending (queue: {in_queue}, progress: {in_progress})")
        
        return results


# ==========================================
# USAGE EXAMPLE
# ==========================================

if __name__ == "__main__":
    # Initialize client
    client = I2VClient(
        api_key="YOUR_RUNPOD_API_KEY",  # Replace with your API key
        endpoint_id="YOUR_ENDPOINT_ID"   # Replace with your endpoint ID
    )
    
    # Define batch of videos to generate
    jobs = [
        {
            "image": "https://parentearn.com/VideoGen/logan-1.png",
            "prompt": "Slow push-in on Logan hammering the bag, camera shake on each punch",
            "resolution": "480p",
            "sample_steps": 20,
            "frame_num": 81  # ~5 seconds
        },
        {
            "image": "https://parentearn.com/VideoGen/scene-2.png",
            "prompt": "Character turns head slowly, eyes tracking movement",
            "resolution": "480p",
            "sample_steps": 20,
            "frame_num": 49  # ~3 seconds
        }
    ]
    
    # Process batch with automatic worker management
    results = client.process_batch(jobs, auto_manage_workers=True)
    
    # Print results
    print("\n" + "="*50)
    print("BATCH COMPLETE")
    print("="*50)
    for i, result in enumerate(results):
        print(f"\nVideo {i+1}:")
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  URL: {result.get('video_url')}")
            print(f"  Time: {result.get('generation_time')} seconds")
```

---

## Performance & Timing

### Single GPU (A100 80GB)

| Phase | Duration |
|-------|----------|
| Worker Startup (cold) | 30-60 seconds |
| Model Load to GPU | 4-5 minutes |
| **Cold Start Total** | **5-6 minutes** |
| Generation (2 sec video, 33 frames) | ~5 minutes |
| Generation (5 sec video, 81 frames) | ~7 minutes |
| Generation (10 sec video, 161 frames) | ~12-15 minutes |

### With Active Worker (Warm)

| Phase | Duration |
|-------|----------|
| Worker Startup | 0 (already running) |
| Model Load | 0 (already in VRAM) |
| Generation (5 sec video) | ~5-7 minutes |

---

## Multi-GPU Considerations

### Will Multi-GPU Speed Up Generation?

**Short Answer:** Limited benefit for single video generation.

**Detailed Analysis:**

| Aspect | Single GPU | Multi-GPU |
|--------|------------|-----------|
| **Single Video** | ~7 min | ~7 min (no speedup) |
| **Concurrent Videos** | 1 at a time | Multiple parallel |
| **Cost** | 1x GPU rate | Nx GPU rate |
| **VRAM per GPU** | 80GB (A100) | 80GB each |

**Why Limited Speedup for Single Video:**

1. **Model Architecture:** Wan2.2 I2V-A14B is designed for single-GPU inference
2. **Tensor Parallelism:** Would require code modifications to split model across GPUs
3. **Communication Overhead:** Multi-GPU coordination adds latency

**When Multi-GPU Helps:**

1. **Parallel Processing:** Process multiple videos simultaneously
2. **Higher Throughput:** Handle more requests concurrently
3. **Batch Processing:** Generate many videos faster overall

**Recommended Multi-GPU Strategy:**

```python
# Instead of 1 worker with 2 GPUs, use 2 workers with 1 GPU each
# This allows parallel video generation

# In RunPod endpoint settings:
# - Max Workers: 2
# - GPUs per Worker: 1

# Then submit multiple jobs - they'll run in parallel
job_ids = []
for video_config in batch:
    job_id = client.generate_video(**video_config)
    job_ids.append(job_id)

# All jobs process concurrently on separate workers
```

**Cost Comparison for 10 Videos:**

| Strategy | Config | Time | Cost |
|----------|--------|------|------|
| Sequential | 1 worker, 1 GPU | ~70 min | ~$2.20 |
| Parallel | 2 workers, 1 GPU each | ~35 min | ~$2.20 |
| Parallel | 5 workers, 1 GPU each | ~14 min | ~$2.20 |

**Conclusion:** Use multiple single-GPU workers for throughput, not multi-GPU for single video speedup.

---

## Frame Count Reference

| Duration | Frames (16 fps) | Recommended For |
|----------|-----------------|-----------------|
| 1 second | 17 | Quick tests |
| 2 seconds | 33 | Short clips |
| 3 seconds | 49 | Standard clips |
| 5 seconds | 81 | Full scenes |
| 7 seconds | 113 | Extended scenes |
| 10 seconds | 161 | Maximum length |

---

## Resolution Options

| Resolution | Dimensions | Quality | Speed | VRAM Usage |
|------------|------------|---------|-------|------------|
| 480p | 832×480 | Good | Faster | ~41 GB |
| 720p | 1280×720 | Better | Slower | ~65 GB |

---

## Sample Steps Guide

| Steps | Quality | Speed | Recommended For |
|-------|---------|-------|-----------------|
| 15 | Acceptable | Fastest | Previews, testing |
| 20 | Good | Fast | Production (balanced) |
| 30 | Better | Medium | High quality |
| 40 | Best | Slow | Maximum quality |
| 50 | Marginal gain | Slowest | Diminishing returns |

---

## Error Handling

```python
def generate_with_retry(client, job_config, max_retries=3):
    """Generate video with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            job_id = client.generate_video(**job_config)
            result = client.wait_for_completion(job_id)
            
            if "error" not in result:
                return result
            
            print(f"Attempt {attempt + 1} failed: {result['error']}")
            
        except Exception as e:
            print(f"Attempt {attempt + 1} exception: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(10)  # Wait before retry
    
    return {"error": "Max retries exceeded"}
```

---

## Rate Limits & Best Practices

1. **Concurrent Jobs:** Limited by max workers in endpoint settings
2. **Timeout:** Set execution timeout to at least 1800 seconds (30 min)
3. **Polling:** Use 5-10 second intervals for status checks
4. **Worker Management:** Disable workers when not in use to save costs

---

## Quick Start

```python
from i2v_client import I2VClient

# Initialize
client = I2VClient(
    api_key="YOUR_API_KEY",
    endpoint_id="zpms3tyvv2gelr"
)

# Single video generation
client.enable_worker()
client.wait_for_worker_ready()

job_id = client.generate_video(
    image="https://example.com/image.png",
    prompt="Camera slowly zooms in",
    frame_num=81
)

result = client.wait_for_completion(job_id)
print(f"Video URL: {result['video_url']}")

client.disable_worker()
```

---

## Support

- **RunPod Documentation:** https://docs.runpod.io/
- **Wan2.2 Model:** https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- **Issues:** https://github.com/romantony/storystudio-wan2-i2v/issues
