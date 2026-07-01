# storystudio-wan2-i2v — Definitive Build Reference

**Status: PRODUCTION STABLE**  
**Confirmed working as of: 2026-07-01**  
**GitHub:** `https://github.com/romantony/storystudio-wan2-i2v`  
**RunPod endpoint:** `nd7wloyvj09xwy`  
**GPU:** NVIDIA RTX 6000 Ada Generation — 47.4 GB GDDR6 ECC  

---

## What This Is

A RunPod serverless worker that runs **Wan2.2 Image-to-Video (I2V) 14B in FP8** precision. Given a reference image and a text prompt, it generates a ~5-second MP4 video (81 frames at 16 fps) in 480p or 720p.

The model runs entirely warm between jobs — both DiT models stay resident on the GPU at all times. Cold start (first job on a fresh worker) includes model load (~184s). Subsequent jobs on the same worker skip loading.

---

## Architecture

### Pipeline

```
User job (RunPod)
    ↓
handler_v2.py          ← RunPod serverless entrypoint
    ↓ Unix socket (/tmp/wan2_model_server.sock)
model_server.py        ← Persistent background process, holds GPU memory
    ↓
WanI2V pipeline        ← Wan2.2 native code (/workspace/wan22/)
    ├── UMT5-XXL text encoder  (t5_cpu=True → CPU; temporarily GPU during encode)
    ├── CLIP vision encoder    (GPU)
    ├── high_noise DiT (14B)   (FP8, GPU resident, ~13.6 GB VRAM)
    ├── low_noise  DiT (14B)   (FP8, GPU resident, ~13.6 GB VRAM)
    └── VAE encoder/decoder    (GPU)
```

### Wan2.2 Native Code

Cloned from `https://github.com/Wan-Video/Wan2.2.git` at Docker build time into `/workspace/wan22/`.  
`wan/__init__.py` is trimmed to I2V-only imports (avoids pulling in WanS2V/decord, WanT2V, WanTI2V, WanAnimate — none needed here).

### Dual-DiT structure

Wan2.2 I2V uses TWO separate WanModel instances:
- `high_noise_model` — handles early diffusion timesteps (noisy frames)  
- `low_noise_model` — handles late diffusion timesteps (refined frames)

Both are WanModel instances (`dim=5120, ffn_dim=13824, num_heads=40, num_layers=40`). `WanI2V` is a pipeline wrapper class, **not** an `nn.Module` — calling `.modules()` on it fails.

---

## Model Weights on Network Volume

**Volume path:** `/runpod-volume/wan22-qfp8/`  
**Source:** `nalexand/Wan2.2-I2V-A14B-FP8` (downloaded once to network volume)  
**Format detected via:** `/runpod-volume/wan22-qfp8/format.json` → `"format": "wan22-i2v-qfp8-v1"`

```
/runpod-volume/wan22-qfp8/
├── format.json                          ← format marker: wan22-i2v-qfp8-v1
├── models_t5_umt5-xxl-enc-bf16.pth     ← UMT5-XXL text encoder (BF16, ~9–10 GB)
├── models_clip_open-clip-xlm-roberta-large-vit-h-14-frozen.pth  ← CLIP vision encoder
├── Wan2.1-VAE.pth                       ← VAE encoder/decoder
├── high_noise_model/
│   ├── config.json                      ← WanModel architecture config
│   └── *.safetensors                    ← per-block FP8 shards (float8_e4m3fn + scale)
└── low_noise_model/
    ├── config.json
    └── *.safetensors
```

### FP8 Format (wan22-i2v-qfp8-v1)

Each safetensors shard contains:
- `some.layer.weight` → `float8_e4m3fn` tensor — stored as-is on GPU
- `some.layer.weight.scale` → BF16 scalar — per-tensor dequant scale
- `some.layer.bias` → BF16 tensor
- Non-linear params (norms, embeddings, conv) → BF16

At load time, each `nn.Linear` in the WanModel skeleton is replaced with an `FP8Linear` instance that stores the FP8 weight + scale and dequantizes on-the-fly per forward pass.

---

## Docker Image

**Base image:** `runpod/pytorch:1.0.7-cu1290-torch260-ubuntu2204`  
**Final PyTorch:** `torch==2.7.0+cu128` (re-pinned during build)  
**FlashAttention:** `flash_attn==2.8.2` (cu12, torch2.7, cxx11abi confirmed active)  

### Key ENV vars set in Dockerfile

```dockerfile
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV MODEL_PATH=/runpod-volume/wan22-qfp8
ENV HF_HOME=/runpod-volume/huggingface
```

`expandable_segments:True` allows the CUDA allocator to grow allocations in-place instead of always reserving contiguous blocks — reduces fragmentation during the diffusion loop.

---

## Critical Build-Time Patches (Dockerfile)

### Patch 1 — rope_apply float64 → float32

**File:** `/workspace/wan22/wan/modules/model.py`  
**Why:** Wan2.2's `rope_apply` internally casts to `torch.float64`, creating `complex128` intermediates. On RTX 6000 Ada (47.4 GB) with both FP8 DiTs resident (~27.2 GB), the float64 path creates:
- `x[i, :seq_len].to(float64)` → 1.30 GB complex128
- `x_i * freqs_i` result → +1.30 GB (simultaneous)
- `.float()` output → +0.62 GB float32  
- **Total peak: ~3.22 GB extra, OOM (only 0.60 GB free)**

**Fix applied:**
```python
# Change 1: float64 → float32 (halves complex intermediate size)
'.to(torch.float64).reshape('  →  '.float().reshape('

# Change 2: force freqs_i to complex64 so multiply stays in float32
'x_i = torch.view_as_real(x_i * freqs_i).flatten(2)'
    →
'x_i = torch.view_as_real(x_i * freqs_i.to(x_i.dtype)).flatten(2)'
```

**Result:** Peak drops to 2 × 0.65 GB = 1.30 GB above base → total ~28.5 GB → fits ✓

### Patch 2 — wan/__init__.py trimmed

Replaced with I2V-only imports to avoid pulling in WanS2V (requires `decord`, not installed):
```python
from . import configs, distributed, modules
from .image2video import WanI2V
```

### Patch 3 — T5 device fix

```bash
sed -i 's/device=torch\.cuda\.current_device()/device=0/g' /workspace/wan22/wan/modules/t5.py
```

Forces T5 to use device index `0` instead of `torch.cuda.current_device()` which can fail in certain init contexts.

### Patch 4 — FlashAttention fallback

If `flash_attn` is unavailable (wheel mismatch), the build patches all `from .attention import flash_attention` imports to redirect to the PyTorch SDPA fallback `attention()`. This is a no-op when FA2 is successfully installed.

---

## model_server.py — Key Settings

### Resident vs offload decision

```python
RESIDENT_VRAM_THRESHOLD_GB = 40
# RTX 6000 Ada: 47.4 GB → keep_resident = True → both DiTs stay on GPU
# RTX 5090 (32 GB) or smaller → keep_resident = False → page one DiT at a time
```

### WanI2V init

```python
WanI2V(
    config=cfg,
    checkpoint_dir=MODEL_PATH,
    device_id=0,
    rank=0,
    t5_fsdp=False,
    dit_fsdp=False,
    use_sp=False,
    t5_cpu=True,       # UMT5 stays in CPU RAM (not GPU) — temporarily moved to GPU during encode
    init_on_cpu=True,  # DiTs start on CPU; _load_our_fp8 moves them to GPU after
    convert_model_dtype=False,
)
```

### generate() call parameters

```python
self.pipe.generate(
    input_prompt=prompt,
    img=image,
    max_area=max_area,       # 832*480=399360 for 480p, 1280*720=921600 for 720p
    frame_num=frame_num,     # 81 frames = ~5s at 16fps
    shift=flow_shift,        # 3.0 for 480p, 5.0 for 720p
    sample_solver='unipc',   # UniPC multi-step solver
    sampling_steps=sample_steps,  # default 15 in test; handler accepts 4–50
    guide_scale=5.0,
    n_prompt=n_prompt,
    seed=-1,                 # random seed each job
    offload_model=offload,   # False on RTX 6000 Ada (keep_resident=True)
)
```

### Dequant cache

```python
reserve_gb = float(os.getenv("DEQUANT_CACHE_RESERVE_GB", "14"))
# On RTX 6000 Ada after model load:
#   free VRAM ≈ 19.7 GB, reserve = 14 GB → cache budget = 5.7 GB
# Cache stores BF16-dequantized copies of FP8 weights after step 0.
# Budget reached → remaining layers dequant per-step as usual.
```

`DEQUANT_CACHE=off` env var disables caching entirely.

### T5 GPU offload (commit 566a474)

The UMT5 text encoder runs on CPU by default (`t5_cpu=True`). This took **218 seconds** per job. Fix: `_instrument_timing()` wraps `T5Encoder.__call__` to temporarily move the model to GPU for each encode call.

```python
# Moves t5_module to cuda:0 before encode, back to cpu after.
# Also patches t5_module.forward to move CPU tokenizer tensors to GPU
# (tokenizer always outputs CPU tensors, causing device mismatch otherwise).
if was_on_cpu:
    t5_module.to(torch.device('cuda', 0))
    _orig_fwd = t5_module.forward
    def _gpu_fwd(*fa, **fk):
        fa = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in fa)
        fk = {k2: v.cuda() if isinstance(v, torch.Tensor) else v for k2, v in fk.items()}
        return _orig_fwd(*fa, **fk)
    t5_module.forward = _gpu_fwd
# After encode:
    t5_module.forward = _orig_fwd
    t5_module.to('cpu')
    torch.cuda.empty_cache()
```

**VRAM during T5 encode:** DiTs (27.2 GB) + UMT5 (~9 GB) = ~36 GB → fits in 47.4 GB ✓

### _clear_dequant_cache

Called after any job failure. Iterates `high_noise_model.modules()` and `low_noise_model.modules()` directly (NOT `self.pipe.modules()` — WanI2V is not nn.Module).

```python
for attr in ('high_noise_model', 'low_noise_model'):
    model = getattr(self.pipe, attr, None)
    for module in model.modules():
        if hasattr(module, '_cached_w') and module._cached_w is not None:
            module._cached_w = None
FP8Linear.cache_used_bytes = 0
torch.cuda.empty_cache()
```

---

## handler_v2.py — Key Settings

### Resolution map

```python
RESOLUTION_MAP = {
    "480p": "832*480",
    "720p": "1280*720"
}
```

### Image download

For URLs containing `R2_PUBLIC_URL` (storyaistudio.app / parentearn.com), the handler fetches directly from R2 via boto3 (bypasses the web server). All other URLs use `requests.get()`.

### R2 upload

Video uploaded to Cloudflare R2 bucket `storystudio`, folder `VideoGen/`.  
Public URL pattern: `https://storyaistudio.app/VideoGen/YYYYMMDD_HHMMSS_i2v_{resolution}_{steps}steps.mp4`

### Input validation

```python
sample_steps: 4–50 (int)
frame_num:    17–161 (int)
resolution:   "480p" | "720p"
image:        URL string (http/https) or base64-encoded bytes
```

---

## VRAM Budget (RTX 6000 Ada, 47.4 GB)

| Component | VRAM | Notes |
|---|---|---|
| high_noise DiT (FP8) | ~13.6 GB | float8_e4m3fn + BF16 non-linear params |
| low_noise DiT (FP8) | ~13.6 GB | same |
| **Total DiTs** | **~27.2 GB** | confirmed in logs |
| UMT5-XXL text encoder | 0 GB resident | CPU RAM, ~9 GB; temporary GPU during encode |
| CLIP vision encoder | ~0.5 GB | GPU |
| VAE | ~0.5 GB | GPU |
| **Total base** | **~28.2 GB** | |
| Dequant cache | 5.7 GB | caches BF16 weights after step 0 |
| Activations (peak) | ~8–10 GB | during diffusion step forward pass |
| **Total peak** | **~44–46 GB** | <47.4 GB ✓ |

**During T5 encode (before diffusion loop):**  
- DiTs (27.2 GB) + UMT5 (9 GB) = 36.2 GB — 11.2 GB headroom ✓  
- Dequant cache is empty at this point (filled during diffusion steps)

---

## Confirmed Performance (2026-07-01 logs)

### Run 1 — baseline (commit 86f1af0, no T5 GPU offload)
- Cold start / model load: **184.5s**
- T5 encode: **218.0s** (UMT5 on CPU)
- VAE encode: **4.1s**
- Diffusion (15 steps): **479.5s** (32.0s/step)
- VAE decode: **7.3s**
- **Total model time: 708.9s**
- Video: 5.1 MB, 480p, 81 frames

### Run 2 — T5 GPU offload active (commit 566a474)
- T5 encode: **~44s** (down from 218s, 5× faster)
- Diffusion (15 steps): **~480s** (unchanged)
- VAE: **~11s** (unchanged)
- **Total model time: 535.5s** — **−173s vs baseline (−24%)**
- Video: 5.07 MB, 480p, 81 frames
- Prompt: Pixar Nativity scene (long, complex)

### Step time
**32.0s/step** with FlashAttention 2.8.2 active + partial dequant cache (~5.7 GB budget, fills during step 0 then partially hits on steps 1–14).

---

## Git Commits (session 2026-07-01)

| Hash | Description |
|---|---|
| `a458144` | Restore reserve=14, fix _clear_dequant_cache (initial version with bug) |
| `3b31e74` | Fix _clear_dequant_cache AttributeError + rope_apply patch (wrong rebind approach) |
| `86f1af0` | **Correct rope_apply fix** (float64→float32, complex64 multiply) — first successful generation |
| `bcad82c` | T5 GPU offload wrapper (initial, had device mismatch bug) |
| `566a474` | **Fix T5 GPU offload**: patch forward to move tokenizer CPU tensors to GPU |

Current production commit: **`566a474`**

---

## Key Bugs Fixed (and Why)

### 1. CUDA OOM in rope_apply (root cause of all early failures)

`rope_apply` in `wan/modules/model.py` used `torch.float64` internally. With 27.2 GB DiTs resident, the float64 path needed 3.22 GB extra VRAM during each call, exceeding the 47.4 GB card. Fixed by patching to float32 in the Dockerfile (two-line patch).

### 2. `WanI2V.modules()` AttributeError

`_clear_dequant_cache` called `self.pipe.modules()` — but `WanI2V` is a pipeline class, not `nn.Module`. Fixed by iterating `self.pipe.high_noise_model.modules()` and `self.pipe.low_noise_model.modules()` directly.

### 3. T5 device mismatch on GPU offload

When we moved `t5_module` to GPU, the tokenizer (running inside `T5Encoder.__call__`) still produced CPU tensors. The first embedding lookup (`index_select(weight_on_GPU, 0, ids_on_CPU)`) crashed. Fixed by temporarily patching `t5_module.forward` to move all tensor args to CUDA before the forward call.

---

## Environment Variables (RunPod endpoint configuration)

| Variable | Value | Notes |
|---|---|---|
| `MODEL_PATH` | `/runpod-volume/wan22-qfp8` | Set in Dockerfile ENV |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduces allocator fragmentation |
| `HF_HOME` | `/runpod-volume/huggingface` | HuggingFace cache on network volume |
| `DEQUANT_CACHE_RESERVE_GB` | `14` (default) | VRAM headroom reserved beyond cache |
| `DEQUANT_CACHE` | `auto` (default) | Set to `off` to disable caching |
| `R2_ACCOUNT_ID` | `620baa808df08b1a30d448989365f7dd` | Set in RunPod endpoint secrets |
| `R2_ACCESS_KEY_ID` | (secret) | Cloudflare R2 API key |
| `R2_SECRET_ACCESS_KEY` | (secret) | Cloudflare R2 secret |
| `R2_BUCKET_NAME` | `storystudio` | |
| `R2_PUBLIC_URL` | `storyaistudio.app` | Public domain for video URLs |

---

## API — Input Payload

```json
{
  "input": {
    "image":        "<URL or base64 string>",
    "prompt":       "Your text prompt describing the desired video motion and style",
    "n_prompt":     "Optional negative prompt (what to avoid)",
    "resolution":   "480p",
    "sample_steps": 15,
    "frame_num":    81
  }
}
```

### Output

```json
{
  "video_url":              "https://storyaistudio.app/VideoGen/YYYYMMDD_HHMMSS_i2v_480p_15steps.mp4",
  "generation_time":        535.42,
  "model_generation_time":  535.49,
  "video_size_mb":          5.07,
  "resolution":             "480p",
  "sample_steps":           15,
  "frame_num":              81
}
```

### Recommended settings

| Use case | resolution | sample_steps | frame_num | Expected time |
|---|---|---|---|---|
| Fast preview | `480p` | 10 | 49 | ~360s |
| Standard | `480p` | 15 | 81 | ~535s |
| High quality | `480p` | 25 | 81 | ~850s |
| 720p standard | `720p` | 15 | 81 | TBD |

---

## What This Build Does NOT Support

- **Text-to-video (T2V)** — I2V only; `image` input is required
- **FLF2V (First-Last Frame)** — separate model variant needed
- **Lightning 4-step** — different distilled weights needed (see `WAN22-I2V-LIGHTNING-4STEP-BUILD.md`)
- **720p on 32 GB cards** — would OOM; requires 40+ GB
- **Multiple concurrent jobs** — single model server, sequential processing

---

## Cold Start Sequence

```
Container start
  ↓ handler_v2.py runs
  ↓ verify_model_present()  — checks for models_t5_umt5-xxl-enc-bf16.pth
  ↓ start_model_server()    — spawns model_server.py subprocess
      ↓ WanI2V.__init__()   — loads T5, CLIP, VAE, builds empty DiT skeletons
      ↓ _load_our_fp8()     — loads FP8 shards, moves both DiTs to GPU
      ↓ _configure_dequant_cache()  — sets cache budget
      ↓ _instrument_timing()        — wraps T5 + VAE calls
      ↓ Unix socket ready
  ↓ handler_v2.py connects to socket → ✓ ready
Total cold start: ~184–200s
```

---

## Next Build

See `WAN22-I2V-LIGHTNING-4STEP-BUILD.md` for the 4-step generation plan using  
`lightx2v/Wan2.2-Distill-Models` (distilled FP8 weights, `guidance_scale=[3.5, 3.5]`, `sample_shift=5.0`).  
Expected total generation time: **~142s (~2m 22s)** per job.
