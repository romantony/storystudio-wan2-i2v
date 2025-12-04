# Wan2.2 I2V (Image-to-Video) RunPod Serverless

Generate animated videos from static images using Wan2.2 I2V-A14B model.

## Features

- **Image-to-Video**: Animate any static image with natural motion
- **Multiple Resolutions**: 480p and 720p support
- **Configurable Length**: 1-10 seconds of video
- **R2 Storage**: Automatic upload to Cloudflare R2

## API Usage

### Generate Video from Image

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {api_key}" \
  -d '{
    "input": {
      "image": "https://example.com/image.png",
      "prompt": "A cat walking gracefully through the scene",
      "resolution": "480p",
      "sample_steps": 30,
      "frame_num": 81
    }
  }'
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | ✅ | - | Image URL or base64 encoded |
| `prompt` | string | ❌ | "" | Motion/action description |
| `resolution` | string | ❌ | "480p" | "480p" or "720p" |
| `sample_steps` | int | ❌ | 30 | 15-50 (higher = better quality) |
| `frame_num` | int | ❌ | 81 | 17-161 frames (~1-10 sec at 16fps) |

### Response

```json
{
  "video_url": "https://parentearn.com/VideoGen/20251204_123456_i2v_480p.mp4",
  "generation_time": 120.5,
  "video_size_mb": 2.5,
  "resolution": "480p",
  "sample_steps": 30,
  "frame_num": 81
}
```

## Frame Number Guide

| frame_num | Duration | Use Case |
|-----------|----------|----------|
| 17 | ~1 sec | Quick preview |
| 49 | ~3 sec | Short clip |
| 81 | ~5 sec | Standard (default) |
| 129 | ~8 sec | Extended |
| 161 | ~10 sec | Maximum length |

## Differences from S2V

| Feature | I2V | S2V |
|---------|-----|-----|
| Audio input | ❌ Not needed | ✅ Required |
| Lip sync | ❌ No | ✅ Yes |
| Use case | General animation | Talking head |
| Motion | Scene-based | Speech-driven |

## Docker Image

```
romantony/wan2-i2v:1.0.0
```

## RunPod Setup

1. Create new Serverless Endpoint
2. Use image: `romantony/wan2-i2v:1.0.0`
3. Select GPU: A100 80GB or H100
4. Attach network volume: `/runpod-volume`
5. Set timeout: 1800 seconds

## Network Volume

Uses same volume as S2V. Shared components:
- `Wan2.1_VAE.pth` (~0.5GB) - Shared ✅
- `models_t5_umt5-xxl-enc-bf16.pth` (~11GB) - Shared ✅
- `diffusion_pytorch_model-*.safetensors` (~33GB) - I2V specific

Total additional storage for I2V: ~33GB
