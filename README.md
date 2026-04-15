# ComfyUI GPU Resident Loader

A ComfyUI custom-node pack for **faster time-to-VRAM**, **selective safetensors loading**, **sticky GPU residency control**, and **visibility into compatible external GPU model caches**.

This repo does three related jobs:

1. **Installs startup-time monkey patches** before any workflow nodes run.
2. Ships **KJ-style resident loader nodes** for diffusion models and checkpoints.
3. Maintains a **live residency system** for native ComfyUI objects and compatible external caches, with preload / pin / evict / report controls for native tracked objects and automatic snapshot / trim / eviction support for compatible external entries.

It is not just a “clean RAM” addon. The main target is the path from model file -> tensors -> live ComfyUI object -> VRAM retention, including GPU-resident caches that live outside ComfyUI’s normal loaded-model list.

## Why this exists

ComfyUI’s default behavior mixes together two separate concerns:

- **ingest path** — where tensors are first materialized while a model is being read, and
- **residency policy** — where the finished model tends to live afterwards.

Those are not the same problem.

This repo focuses on both:

- For **`.safetensors`**, it tries to keep eligible loads on the narrowest, most GPU-friendly path it can.
- For **resident diffusion and checkpoint-model loads**, it avoids broad checkpoint materialization by selecting only the detected UNet keys where possible.
- For **runtime VRAM pressure**, it adds a sticky-priority registry and teaches ComfyUI’s unload path to protect higher-value resident entries until enough VRAM must be reclaimed.
- For **compatible external GPU caches** that bypass `comfy.model_management.current_loaded_models`, it can discover supported providers at runtime and include their entries in snapshot and trim decisions.
- For **manual control**, it exposes nodes that let you preload, pin, evict, and inspect tracked native models, CLIPs, and VAEs.

## What changes at startup

`__init__.py` calls `startup.install_patches()`, which applies the core monkey patches exactly once when the custom node is imported.

Those startup patches cover the built-in ComfyUI load and residency paths below. Compatible external-cache integrations are installed lazily later, on demand, when a supported module is actually present in the running process.

### Patched functions / methods

Current patch surface:

- `comfy.utils.load_torch_file`
- `comfy.clip_vision.load_torch_file` (redirected to the patched `comfy.utils.load_torch_file` when present)
- `comfy.model_management.free_memory`
- `comfy.model_management.load_models_gpu`
- `comfy.model_management.unet_offload_device`
- `comfy.model_management.text_encoder_offload_device`
- `comfy.model_management.vae_offload_device`
- `comfy.model_management.text_encoder_device`
- `comfy.model_management.vae_device`
- `comfy.model_management.unet_inital_load_device`
- `comfy.model_management.LoadedModel.model_unload`
- `comfy.model_patcher.ModelPatcher.detach`
- `comfy.sd.load_checkpoint_guess_config`
- `comfy.sd.load_diffusion_model`
- `comfy.sd.load_clip`
- `comfy.sd.VAE.encode`
- `comfy.sd.VAE.decode`
- `comfy.clip_vision.load`
- `comfy.controlnet.load_controlnet`
- `comfy.diffusers_load.load_diffusers`

### Lazily installed external integrations

Current external integration surface:

- compatible **SeedVR2** `src/core/model_cache.py` modules discovered at runtime

When a compatible SeedVR2 cache module is present, the repo wraps:

- `GlobalModelCache.set_dit`
- `GlobalModelCache.set_vae`
- `GlobalModelCache.replace_dit` (when that method exists in the installed SeedVR2 build)
- `GlobalModelCache.replace_vae` (when that method exists in the installed SeedVR2 build)
- `GlobalModelCache.remove_dit`
- `GlobalModelCache.remove_vae`

That lazy integration lets the loader:

- mirror SeedVR2-owned cached **DiT** and **VAE** objects into a separate external residency registry
- refresh byte / device / claimed-state metadata from the live cached object
- evict those entries through SeedVR2’s own removal path instead of assuming they live in `comfy.model_management.current_loaded_models`

## What those patches do

### 1) `load_torch_file` becomes residency-aware

The patched loader:

- detects the active load context (`model`, `clip`, `vae`, `checkpoint`, etc.)
- picks an explicit GPU target device when the active policy wants GPU ingest
- attempts **direct safetensors reads** on the requested device
- falls back to **CPU read + tensor-by-tensor copy** if direct GPU safetensors loading fails
- still uses **CPU-first `torch.load()`** for pickle formats (`.ckpt`, `.pt`, `.pth`, `.bin`)
- records the actual load method in the residency registry

For safetensors loads happening inside a `model` / `clip` / `vae` context, it can select only the detected component keys from the file header instead of pulling the full file into memory first.

### 2) Loader contexts are attached to stock ComfyUI load paths

These stock paths are wrapped with registry context and output binding:

- checkpoint loads
- diffusion-model loads
- CLIP loads
- CLIP Vision loads
- ControlNet loads
- diffusers loads

That means the native registry is not limited to the custom resident nodes. Stock ComfyUI loaders that pass through these paths are also tracked.

### 3) Compatible external caches can join the residency system lazily

When a compatible SeedVR2 cache module is present, the loader installs cache-level hooks that register SeedVR2-owned cached DiT / VAE objects into a separate external registry.

Those entries:

- are refreshed from the live cached object at runtime
- appear in **Registry Snapshot** output under `external_entries`
- are considered by load-scoped VRAM trimming even though they are not part of `current_loaded_models`
- are evicted through SeedVR2’s own cache-removal methods rather than the normal Comfy unload path

### 4) Device/offload policy is overridden

Depending on the active policy, the patcher can steer:

- initial UNet load device
- CLIP/Text Encoder device
- VAE device
- offload devices for UNet / CLIP / VAE

This is how `prefer_gpu` and `sticky_gpu` keep more of the hot path on the GPU side than stock ComfyUI would.

### 5) `free_memory()` becomes sticky-aware

Under `sticky_gpu`, `comfy.model_management.free_memory()` is patched so that:

- sticky tracked wrappers are considered first
- higher-priority sticky entries are protected first
- lower-priority or older sticky entries yield first when VRAM must be reclaimed
- a transient protection floor is applied so ComfyUI does not immediately tear down high-value resident entries for small requests

### 6) Clone replacement is hardened

`load_models_gpu()` is patched to fully unload clone-conflict wrappers before replacement instead of relying on a shallow detach path that can leave base weights patched.

### 7) Unload / detach is redirected to CPU when needed

`LoadedModel.model_unload()` and `ModelPatcher.detach()` are patched so that unloads which would otherwise not reclaim VRAM are redirected through a CPU offload target first.

### 8) VAE encode/decode gets a sticky-safe path

Under `sticky_gpu`, patched `VAE.encode()` and `VAE.decode()`:

- cap the working batch count when necessary to preserve transient VRAM headroom
- retry with tiled VAE encode/decode on OOM

That behavior is not a general performance feature toggle. It exists to reduce avoidable VRAM spikes while sticky residency is active.

## Policies

The registry exposes four global policies:

### `legacy`

Stay closest to stock ComfyUI behavior. Registry tracking still exists, but the patcher does not aggressively steer ingest/offload toward the GPU path.

### `balanced`

Keep registry tracking and diagnostics without aggressive GPU residency behavior.

### `prefer_gpu`

Prefer GPU ingest for tracked model-like loads and keep the faster side of the device/offload policy for:

- diffusion models
- CLIP / text encoders
- ControlNets

This policy does **not** auto-pin tracked objects.

### `sticky_gpu`

Builds on `prefer_gpu` and additionally:

- auto-pins newly bound **models** and **CLIPs**
- keeps **VAE offload** on the GPU side as well
- patches `free_memory()` to protect sticky tracked wrappers by priority
- uses the sticky-safe VAE encode/decode behavior

### Default policy selection

Selection order is:

1. `COMFYUI_GPU_RESIDENT_POLICY`, if set to a supported value
2. `sticky_gpu` when ComfyUI is started with `--gpu_only`
3. `sticky_gpu` when ComfyUI is started with `--highvram`
4. otherwise `prefer_gpu`

Supported values are:

- `legacy`
- `balanced`
- `prefer_gpu`
- `sticky_gpu`

## Included nodes

All nodes live under the `GPU Resident Loader` category.

### Loader nodes

#### Diffusion Model Selector Resident

Returns an absolute path string for a selected diffusion model.

Notes:

- resolves from `diffusion_models`
- also exposes `text_encoders` entries whose filename contains `connector`

#### Diffusion Model Loader Resident

KJ-style diffusion-model loader with these controls:

- `weight_dtype`
- `compute_dtype`
- `patch_cublaslinear`
- `sage_attention`
- `enable_fp16_accumulation`
- optional `extra_state_dict`
- optional `policy_override`

Behavior:

- for `.safetensors`, it loads only the detected UNet portion of the file
- if `extra_state_dict` is provided, only matching UNet keys are merged
- repeated loads reuse a live equivalent model when the source path and loader-relevant options still match
- before GPU-bound loads, it estimates the upcoming footprint and trims only enough lower-priority residency to cover the request plus adaptive headroom

#### Checkpoint Loader Resident

Full checkpoint loader that returns:

- `MODEL`
- `CLIP`
- `VAE`

Behavior:

- shares the same tuning knobs as the resident diffusion-model loader for the model component
- reuses already-live equivalent components when possible
- composes the final output from model / clip / vae component loaders instead of always rebuilding the whole checkpoint path from scratch

#### Checkpoint Model Loader Resident

Model-only checkpoint loader.

Behavior:

- takes the same selective safetensors UNet fast path as the diffusion-model loader
- reuses a live equivalent model when available
- uses the same dtype / attention / cublas / fp16-accumulation knobs as the full checkpoint loader

#### Checkpoint Clip Loader Resident

CLIP-only checkpoint loader.

Behavior:

- can reuse a live equivalent CLIP object
- avoids rebuilding the diffusion model and VAE outputs when only CLIP is needed

#### Checkpoint VAE Loader Resident

VAE-only checkpoint loader.

Behavior:

- can reuse a live equivalent VAE object
- avoids rebuilding the diffusion model and CLIP outputs when only VAE is needed

### Residency nodes

#### Set Global Residency Policy

Sets the active global policy and returns it as a `STRING`.

The loader nodes also expose an optional `policy_override` string input for one-off loads.

#### Registry Snapshot

Returns a composite formatted JSON snapshot with:

- `policy` for the active global policy
- `entries` for native Comfy-managed registry entries
- `external_entries` for compatible external cache entries discovered at runtime

#### Pin Model Residency / Pin CLIP Residency / Pin VAE Residency

Marks a tracked native object as sticky or non-sticky and optionally changes its priority.

#### Preload Model To GPU / Preload CLIP To GPU / Preload VAE To GPU

Calls `load_models_gpu(..., force_full_load=True)` for the selected native object, then updates sticky state / priority in the registry.

#### Evict Model From GPU / Evict CLIP From GPU / Evict VAE From GPU

Attempts to unload the selected native object from the current loaded-model set.

`unpatch_weights=True` performs a full unload path. When eviction succeeds, the node returns `evicted`; otherwise `not_loaded`.

#### Report Model Residency / Report CLIP Residency / Report VAE Residency

Returns a JSON report for a single tracked native object.

If the object is not currently bound in the registry, the node returns a JSON payload with `tracked: false`.

## Adaptive trimming before resident loads

The resident loaders now do load-scoped VRAM trimming themselves.

Before a GPU-bound resident load, the loader estimates required bytes from:

- the safetensors header when possible
- the detected checkpoint component subset when possible
- otherwise the source file size as a fallback

It then requests enough free VRAM for:

- the estimated load size
- adaptive headroom

Current adaptive headroom policy:

- ratio: `12.5%` of the estimated load
- floor: `256 MiB`
- ceiling: `1 GiB`

The trim path prefers to:

- unload non-sticky entries first
- then lower-priority sticky entries
- preserve explicitly kept models
- use partial unload where available
- include compatible external cache entries in the same candidate search when they are visible, on the same device, and not currently claimed/in use

This logic lives in the resident loader path. You do not need a separate “target free VRAM” node for it.

## Registry and observability

The snapshot now exposes two collections:

- `entries` for native Comfy-managed tracked objects
- `external_entries` for compatible external cache objects

### Native registry entries

The native registry tracks residency metadata for bound objects.

Typical per-entry fields include:

- `entry_id`
- `kind`
- `source_path`
- `basename`
- `sticky`
- `priority`
- `created_at`
- `last_touched`
- `loaded_bytes`
- `total_bytes`
- `load_device`
- `offload_device`
- `current_device`
- `last_method`
- `last_report`
- `loader_key`
- `notes`
- `alive`

The `last_method` / `last_report` fields let you see whether a load actually used:

- direct safetensors GPU ingest
- safetensors CPU -> CUDA fallback
- safetensors component-only load
- CPU-first `torch.load()` compatibility path
- a recorded load failure

### External registry entries

Typical external-entry fields include:

- `entry_id`
- `cache_key`
- `kind`
- `source_path`
- `basename`
- `sticky`
- `priority`
- `created_at`
- `last_touched`
- `loaded_bytes`
- `total_bytes`
- `load_device`
- `offload_device`
- `current_device`
- `claimed`
- `notes`
- `alive`
- `external`

For SeedVR2-backed entries, `claimed: true` means the cache object is currently marked in use and is skipped by the automatic trim candidate search.

## What gets tracked

### Native tracked/bound paths

Native tracked paths include:

- resident node loads from this repo
- stock checkpoint loads
- stock diffusion-model loads
- stock CLIP loads
- stock CLIP Vision loads
- stock diffusers loads

ControlNet loads also participate in the patched load context and device-policy path, but this repo does not currently expose dedicated ControlNet residency nodes.

### External tracked paths

Current external integration coverage is:

- SeedVR2 global cached **DiT** entries
- SeedVR2 global cached **VAE** entries

Those entries are discovered lazily from compatible SeedVR2 cache modules at runtime. They are tracked separately from the native registry and participate in snapshot, load-scoped trim, and provider-specific eviction decisions.

## Important limits and non-goals

### Best path is still `.safetensors`

The narrow fast path is built around `.safetensors`.

That is where this repo can:

- inspect headers cheaply
- select only model / clip / vae subsets
- estimate component bytes more accurately
- attempt direct device-targeted reads

### `.ckpt` / `.pt` / pickle formats are still CPU-first

For pickle-based formats, PyTorch still goes through `torch.load()` on CPU first.

The repo can still:

- track those loads
- keep the resulting live objects resident
- reuse equivalent live objects later

It does **not** claim direct-to-GPU ingest for those formats.

### Cross-process persistence is out of scope

This repo does **not** keep VRAM allocations alive after ComfyUI, Python, or WSL exits.

CUDA memory lifetime is process/context scoped. True persistence across process shutdown would need a separate long-lived keeper process or service that owns the CUDA context.

### External integrations are compatibility-based, not universal

The external registry does **not** automatically manage every third-party cache.

At the moment, the documented external integration target is **SeedVR2**. Other custom nodes with private caches remain invisible until this repo grows a provider-specific integration for them.

### It does not automatically capture arbitrary custom loader implementations

The native registry only sees objects that pass through the patched ComfyUI load paths or through this repo’s resident nodes.

If another custom node loads models through its own private code path and bypasses those patched entry points, that object may never become a tracked native registry entry. In that case, the preload / pin / evict / report nodes from this repo cannot manage it until that external loader is integrated or patched.

Likewise, even for supported external providers such as SeedVR2, the current external integration is about **observation + automatic trim/eviction**. This repo does **not** yet expose dedicated external preload / pin / report / evict nodes for provider-owned cache entries.

## Installation

Clone into `custom_nodes`:

```bash
git clone https://github.com/xmarre/ComfyUI-GPU-Resident-Loader ComfyUI/custom_nodes/ComfyUI-GPU-Resident-Loader
```

Install dependencies into the same Python environment ComfyUI uses:

```bash
pip install -r ComfyUI/custom_nodes/ComfyUI-GPU-Resident-Loader/requirements.txt
```

Requirements declared by the repo:

- Python `>=3.10`
- `safetensors>=0.4.3`

Optional SageAttention dependencies are **not** installed by default. Install those separately if you plan to use a SageAttention mode in the resident loaders.

## Basic usage patterns

### 1) Large-VRAM, mostly resident workflow

Recommended baseline:

- start ComfyUI with `--highvram` or set policy manually to `sticky_gpu`
- prefer `.safetensors` for hot models
- load diffusion models through **Diffusion Model Loader Resident**
- use **Preload ... To GPU** for models you know you will reuse
- inspect with **Report ... Residency** or **Registry Snapshot**

### 2) Full checkpoint workflow

Use **Checkpoint Loader Resident** when you want `MODEL + CLIP + VAE` together.

That path can reuse already-live components instead of always rebuilding all three outputs.

### 3) Staged checkpoint workflow

Use component loaders when the graph does not need the whole checkpoint at once:

- **Checkpoint Model Loader Resident** for diffusion model only
- **Checkpoint Clip Loader Resident** for CLIP only
- **Checkpoint VAE Loader Resident** for VAE only

### 4) Manual native residency control

Use:

- **Pin ... Residency** to mark a tracked native entry sticky / non-sticky
- **Preload ... To GPU** to force a full live native load now
- **Evict ... From GPU** to unload it from the current native loaded-model set

### 5) Mixed workflows with SeedVR2 external caching

If SeedVR2 keeps DiT or VAE models in its own global cache, those objects can now show up in **Registry Snapshot** under `external_entries`.

That means:

- you can see that those bytes exist even though they are outside `current_loaded_models`
- resident loader trim can reclaim them automatically when they are visible, on the same device, and not currently claimed/in use
- eviction goes through SeedVR2’s own cache-removal path instead of a normal Comfy wrapper unload

## Notes on compatibility and migration

### Legacy wiring: `extra_state_dict` used as a policy string

The resident diffusion-model loader contains a compatibility shim for older graphs:

- if `extra_state_dict` receives one of the known policy names
- and that value is **not** an existing file path
- it is interpreted as `policy_override` instead

New graphs should connect policy strings to **`policy_override`**, not to `extra_state_dict`.

### Convert hot pickle checkpoints to safetensors

`scripts/convert_checkpoint_to_safetensors.py` is included for one-time conversion of hot `.ckpt` / `.pt` / `.pth` style checkpoints.

Example:

```bash
python ComfyUI/custom_nodes/ComfyUI-GPU-Resident-Loader/scripts/convert_checkpoint_to_safetensors.py \
  --input /path/to/model.ckpt \
  --output /path/to/model.safetensors
```

Optional flags:

- `--state-dict-key <key>` to extract a different top-level dict key
- `--allow-non-tensor-values` to skip non-tensor entries instead of failing

## License

GPL-3.0-or-later.

This repo stays GPL-compatible because it adapts behavior from GPL-licensed ComfyUI and mirrors relevant loader behavior from GPL-3.0-licensed KJNodes.
