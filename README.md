# ComfyUI GPU Resident Loader

A ComfyUI custom-node pack that targets **time-to-VRAM** and **sticky GPU residency**, not just lower peak host RAM.

It does two things:

1. **Installs startup-time loader and residency patches** before any workflow nodes run.
2. Ships **KJ-compatible loader nodes** for diffusion models and checkpoints, plus preload/pin/evict/report nodes for manual residency control.

## Why this exists

Stock ComfyUI makes separate decisions for:

- where a model **lives after load**, and
- where checkpoint tensors are **materialized first**.

Those are not the same thing.

This repo targets the second problem directly for `.safetensors` by steering eligible loads toward direct GPU ingest, narrowing resident diffusion-model loads down to UNet tensors only, then targets the first problem by overriding offload policy and by teaching `free_memory()` to protect high-priority sticky entries until the VRAM budget says otherwise.

## What is included

### Startup patcher

Installed automatically from `__init__.py` when the custom node loads.

Current patch surface:

- `comfy.utils.load_torch_file`
- `comfy.model_management.free_memory`
- `comfy.model_management.load_models_gpu`
- `comfy.model_management.unet_offload_device`
- `comfy.model_management.text_encoder_offload_device`
- `comfy.model_management.vae_offload_device`
- `comfy.model_management.text_encoder_device`
- `comfy.model_management.vae_device`
- `comfy.model_management.unet_inital_load_device`
- `comfy.sd.load_checkpoint_guess_config`
- `comfy.sd.load_diffusion_model`
- `comfy.sd.load_clip`
- `comfy.clip_vision.load`
- `comfy.controlnet.load_controlnet`
- `comfy.diffusers_load.load_diffusers`

### Loader nodes

- **Diffusion Model Loader Resident**
- **Checkpoint Loader Resident**
- **Checkpoint Model Loader Resident**
- **Checkpoint Clip Loader Resident**
- **Checkpoint VAE Loader Resident**
- **Diffusion Model Selector Resident**

`Diffusion Model Loader Resident` mirrors the relevant KJ diffusion-loader feature surface:

- weight dtype override
- compute dtype override
- cublas-ops toggle
- SageAttention override
- fp16 accumulation toggle
- optional extra-state-dict merge

On `.safetensors`, the resident diffusion-model path now reads only the detected UNet keys and merges only matching keys from any extra state dict. Repeated resident loads also reuse a live equivalent object when the source path and loader-relevant options still match.

Before a GPU-bound resident load needs fresh VRAM, the loader now estimates the upcoming footprint from the model or checkpoint file itself and trims only enough lower-priority residency to cover that request plus a small internal headroom margin. The existing sticky ordering and partial-unload preference stay intact. That decision lives in the loader path, not in a separate user-configured VRAM target node.

### Residency nodes

- **Set Global Residency Policy**
- **Registry Snapshot**
- **Pin Model/CLIP/VAE Residency**
- **Preload Model/CLIP/VAE To GPU**
- **Evict Model/CLIP/VAE From GPU**
- **Report Model/CLIP/VAE Residency**

## Policies

The startup patcher exposes four policies:

- `legacy` — leave ingest/offload behavior close to stock ComfyUI.
- `balanced` — keep the registry and diagnostics, but do not aggressively steer ingest to GPU.
- `prefer_gpu` — prefer GPU ingest, keep UNet/ControlNet/CLIP on the faster side of the device policy, but do not auto-pin tracked objects.
- `sticky_gpu` — prefer GPU ingest, auto-pin the highest-value tracked outputs, and let lower-priority sticky entries yield first when VRAM pressure rises.

Default selection order:

1. `COMFYUI_GPU_RESIDENT_POLICY` environment variable, if set.
2. `sticky_gpu` when `--gpu_only` is active.
3. `sticky_gpu` when `--highvram` is active.
4. otherwise `prefer_gpu`.

## Important scope limits

### Best path: `.safetensors`

This repo is optimized around `.safetensors`.

Direct GPU ingest is attempted for `.safetensors` loads. Resident diffusion-model loads take a header-first selective path and fetch only the detected UNet tensors instead of loading the whole file and filtering later. If the direct path fails, the patcher falls back to CPU read + GPU copy and records that fallback in the registry.

### `.ckpt` / `.pt` remain CPU-first under PyTorch

Those formats still go through `torch.load()`. The repo tracks that path and can still keep the resulting model hot in VRAM, but it does **not** claim true direct-to-GPU checkpoint ingest for pickle-based formats. Resident checkpoint nodes now warn about this when a GPU-resident policy is active so the compatibility path is not mistaken for the fast path.

Use the included conversion helper to migrate hot models to `.safetensors`.

### Cross-process persistence is out of scope

This repo does **not** keep VRAM contents alive after ComfyUI or WSL exits. CUDA memory lifetime is process/context scoped. Achieving persistence across process shutdown requires a long-lived keeper process or server that owns the CUDA context.

## Installation

Clone into `custom_nodes`:

```bash
git clone https://github.com/xmarre/ComfyUI-GPU-Resident-Loader ComfyUI/custom_nodes/ComfyUI-GPU-Resident-Loader
```

Install dependencies inside the same Python environment ComfyUI uses:

```bash
pip install -r ComfyUI/custom_nodes/ComfyUI-GPU-Resident-Loader/requirements.txt
```

Optional SageAttention dependencies are **not** installed by default. Install those separately if you plan to use the SageAttention loader modes.

## Basic usage

### For direct diffusion-model loading

Use **Diffusion Model Loader Resident**.

Recommended on a large VRAM machine:

- policy: `sticky_gpu`
- model format: `.safetensors`
- preload with **Preload Model To GPU**
- inspect with **Report Model Residency** or **Registry Snapshot**

### For full checkpoints

Use **Checkpoint Loader Resident**.

That tracks and binds the resulting diffusion model, CLIP, and VAE independently so they appear in the registry snapshot. If an equivalent live model, CLIP, or VAE already exists, the loader reuses it instead of rebuilding it.

### For staged checkpoint loads

Use:

- **Checkpoint Model Loader Resident** when the workflow only needs the diffusion model
- **Checkpoint Clip Loader Resident** when the workflow only needs the text encoder
- **Checkpoint VAE Loader Resident** when the workflow only needs the VAE

The model-only checkpoint node takes the same selective safetensors UNet fast path as the diffusion-model loader. CLIP-only and VAE-only nodes still use ComfyUI's checkpoint construction logic, but they avoid materializing the other outputs and can reuse an already-live equivalent object.

### For manual residency control

- use **Pin ... Residency** to mark a tracked object sticky or evictable
- use **Preload ... To GPU** to fully materialize it in VRAM immediately
- use **Evict ... From GPU** to unload it from the current loaded-model set

## Observability

Every tracked load stores:

- source path
- last load method
- requested device
- actual device
- sticky flag
- current loaded bytes
- total bytes
- current/offload/load device

That data is surfaced through the report nodes and the registry snapshot node.

## Conversion helper

`scripts/convert_checkpoint_to_safetensors.py` is included for one-time conversion of hot `.ckpt` / `.pt` files into `.safetensors`.

Example:

```bash
python ComfyUI/custom_nodes/ComfyUI-GPU-Resident-Loader/scripts/convert_checkpoint_to_safetensors.py \
  --input /path/to/model.ckpt \
  --output /path/to/model.safetensors
```

## License

GPL-3.0-or-later.

This repo intentionally stays GPL-compatible because it adapts behavior from GPL-licensed ComfyUI and mirrors feature behavior from the GPL-3.0-licensed KJNodes diffusion loader.
