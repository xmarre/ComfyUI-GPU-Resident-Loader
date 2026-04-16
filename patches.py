from __future__ import annotations

import contextlib
import functools
import inspect
import json
import logging
import os
import struct
import threading
from typing import Any, Callable

import torch
from safetensors import safe_open

from .cleanup import (
    _device_matches as _shared_device_matches,
    _should_force_cpu_offload,
    adaptive_headroom_bytes,
    trim_resident_vram,
    unload_loaded_model,
)
from .external_residency import EXTERNAL_REGISTRY, external_objects_for_models, external_trim_enabled
from .residency import (
    KIND_CHECKPOINT,
    KIND_CLIP,
    KIND_CLIP_VISION,
    KIND_CONTROLNET,
    KIND_MODEL,
    KIND_VAE,
    REGISTRY,
)


_LOG = logging.getLogger(__name__)
_PATCHED = False
_ORIGINALS: dict[str, Callable[..., Any]] = {}
_METADATA_CPU_KEY_SUFFIXES = ("spiece_model", "tekken_model", "comfy_quant")
_UNET_PREFIX_CANDIDATES = ("model.diffusion_model.", "model.model.", "net.")
_WARNED_PICKLE_GPU_PATHS: set[str] = set()
_SAFE_TENSORS_COMPONENT_CACHE_MAX = 32
_STICKY_PROTECTION_VRAM_FLOOR_RATIO = 0.125
_STICKY_PROTECTION_VRAM_FLOOR_CEIL_BYTES = 16 * 1024 ** 3
_TILED_VAE_MEMORY_LOCK_ATTR = "_gpu_resident_loader_tiled_memory_lock"
_TILED_VAE_LOCK_INIT = threading.Lock()
_SAFETENSORS_DTYPE_MAP = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "U16": getattr(torch, "uint16", torch.int32),
    "I32": torch.int32,
    "U32": getattr(torch, "uint32", torch.int64),
    "I64": torch.int64,
    "U64": getattr(torch, "uint64", torch.int64),
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3FN": getattr(torch, "float8_e4m3fn", torch.float16),
    "F8_E5M2": getattr(torch, "float8_e5m2", torch.float16),
}


def _safetensors_header_cache_key(path: str) -> tuple[str, int, int]:
    stat = os.stat(path)
    return (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)


def _read_safetensors_header(path: str) -> tuple[dict[str, dict[str, Any]], dict[str, str] | None]:
    with open(path, "rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))

    metadata = header.get("__metadata__")
    tensor_headers = {
        key: value
        for key, value in header.items()
        if key != "__metadata__" and isinstance(value, dict)
    }
    return tensor_headers, metadata if isinstance(metadata, dict) else None


def _torch_dtype_from_safetensors_code(code: str | None) -> torch.dtype:
    if code is None:
        return torch.float32
    return _SAFETENSORS_DTYPE_MAP.get(code, torch.float32)


def _tensor_nbytes_from_header(
    tensor_info: dict[str, Any],
    *,
    dtype_override: torch.dtype | None = None,
) -> int:
    dtype = dtype_override if dtype_override is not None else _torch_dtype_from_safetensors_code(tensor_info.get("dtype"))
    numel = 1
    for dim in tensor_info.get("shape", ()):
        numel *= int(dim)
    return int(numel) * int(torch.empty((), dtype=dtype).element_size())


def _build_meta_state_dict_from_header(tensor_headers: dict[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
    meta_state_dict: dict[str, torch.Tensor] = {}
    for key, tensor_info in tensor_headers.items():
        shape = tuple(int(dim) for dim in tensor_info.get("shape", ()))
        meta_state_dict[key] = torch.empty(
            shape,
            dtype=_torch_dtype_from_safetensors_code(tensor_info.get("dtype")),
            device="meta",
        )
    return meta_state_dict


@functools.lru_cache(maxsize=_SAFE_TENSORS_COMPONENT_CACHE_MAX)
def _cached_component_key_maps(cache_key: tuple[str, int, int]) -> dict[str, Any]:
    import comfy.sd as comfy_sd

    path = cache_key[0]
    tensor_headers, metadata = _read_safetensors_header(path)
    all_keys = tuple(tensor_headers)
    unet_prefix = infer_unet_prefix_from_keys(all_keys)
    meta_state_dict = _build_meta_state_dict_from_header(tensor_headers)
    model_config = comfy_sd.model_detection.model_config_from_unet(meta_state_dict, unet_prefix, metadata=metadata)

    def select_prefixed(prefixes: tuple[str, ...] | list[str] | None) -> tuple[tuple[str, str], ...]:
        if not prefixes:
            return ()
        return tuple(
            (key, key)
            for key in all_keys
            if any(key.startswith(prefix) for prefix in prefixes)
        )

    return {
        "metadata": metadata,
        "unet_prefix": unet_prefix,
        "model_config": model_config,
        "model": tuple((key, key[len(unet_prefix):]) for key in all_keys if key.startswith(unet_prefix)),
        "clip": select_prefixed(getattr(model_config, "text_encoder_key_prefix", None) or ()),
        "vae": select_prefixed(getattr(model_config, "vae_key_prefix", None) or ()),
    }


def checkpoint_component_info_from_header(path: str) -> dict[str, Any] | None:
    try:
        return _cached_component_key_maps(_safetensors_header_cache_key(path))
    except Exception as exc:
        _LOG.warning("GPU Resident Loader: failed to build selective safetensors header map for %s: %s", path, exc)
        return None


def estimate_safetensors_tensor_bytes(
    path: str,
    *,
    selected_keys: list[str] | tuple[str, ...] | set[str] | None = None,
    dtype_override: torch.dtype | None = None,
) -> int | None:
    try:
        tensor_headers, _ = _read_safetensors_header(path)
    except Exception as exc:
        _LOG.warning("GPU Resident Loader: failed to estimate safetensors tensor bytes for %s: %s", path, exc)
        return None

    selected = None if selected_keys is None else set(selected_keys)
    total = 0
    matched = 0
    for key, tensor_info in tensor_headers.items():
        if selected is not None and key not in selected:
            continue
        total += _tensor_nbytes_from_header(tensor_info, dtype_override=dtype_override)
        matched += 1
    if selected is not None and selected and matched == 0:
        _LOG.warning(
            "GPU Resident Loader: selected tensor keys were provided but none matched %s; "
            "treating size as unknown for fallback",
            path,
        )
        return None
    return int(total)


def estimate_checkpoint_component_bytes(
    path: str,
    kind: str,
    *,
    dtype_override: torch.dtype | None = None,
) -> int | None:
    component_maps = checkpoint_component_info_from_header(path)
    if component_maps is None:
        return None
    pairs = component_maps.get(kind, ())
    if not pairs:
        return None
    return estimate_safetensors_tensor_bytes(
        path,
        selected_keys=[source_key for source_key, _ in pairs],
        dtype_override=dtype_override,
    )


def _selected_component_keys_from_header(path: str, kind: str) -> dict[str, str] | None:
    if kind not in {KIND_MODEL, KIND_CLIP, KIND_VAE}:
        return None
    component_maps = checkpoint_component_info_from_header(path)
    if component_maps is None:
        return None

    pairs = component_maps.get(kind, ())
    return dict(pairs) if pairs else None


def _selected_component_suffix(kind: str | None) -> str | None:
    return {
        KIND_MODEL: "model_only",
        KIND_CLIP: "clip_only",
        KIND_VAE: "vae_only",
    }.get(kind)


def _normalize_device(device: Any | None) -> torch.device | None:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    try:
        return torch.device(device)
    except Exception:
        return None


def _device_string(device: torch.device | None) -> str:
    if device is None:
        return "auto"
    return str(device)


def _devices_match(device_a: Any | None, device_b: Any | None) -> bool:
    return _shared_device_matches(device_a, device_b)


def _cpu_offload_required(model: Any, loaded_device: Any | None) -> bool:
    current_device = None
    if hasattr(model, "current_loaded_device"):
        try:
            current_device = _normalize_device(model.current_loaded_device())
        except Exception:
            current_device = None
    if current_device is None:
        current_device = _normalize_device(loaded_device)
    return _should_force_cpu_offload(model, active_device=current_device)


@contextlib.contextmanager
def _temporary_offload_device(model: Any, target_device: torch.device | None):
    if model is None or target_device is None or not hasattr(model, "offload_device"):
        yield False
        return

    original_device = getattr(model, "offload_device", None)
    if _devices_match(original_device, target_device):
        yield False
        return

    setattr(model, "offload_device", target_device)
    try:
        yield True
    finally:
        setattr(model, "offload_device", original_device)


def _safe_open_device_arg(device: torch.device) -> Any:
    if device.type == "cuda":
        return device.index if device.index is not None else torch.cuda.current_device()
    if device.type == "cpu":
        return "cpu"
    return device.type


def _copy_tensor_if_needed(
    tensor: torch.Tensor,
    target_device: torch.device,
    *,
    force_copy: bool = False,
) -> torch.Tensor:
    if tensor.device == target_device and not force_copy:
        return tensor
    return tensor.to(device=target_device, copy=True)


def _tensor_key_requires_cpu(key: str) -> bool:
    return key.endswith(_METADATA_CPU_KEY_SUFFIXES)


def _prepare_loaded_tensor(
    key: str,
    tensor: torch.Tensor,
    requested_device: torch.device,
    *,
    disable_mmap: bool,
    move_to_requested_device: bool = False,
) -> torch.Tensor:
    if _tensor_key_requires_cpu(key):
        return _copy_tensor_if_needed(tensor, torch.device("cpu"))

    if move_to_requested_device and tensor.device != requested_device:
        return _copy_tensor_if_needed(tensor, requested_device)

    if disable_mmap and tensor.device.type == "cpu":
        return _copy_tensor_if_needed(tensor, requested_device, force_copy=True)

    return tensor


def _state_dict_device_summary(sd: dict[str, Any], requested_device: torch.device) -> str:
    devices: set[str] = set()
    for value in sd.values():
        if torch.is_tensor(value):
            devices.add(str(value.device))
    if not devices:
        return str(requested_device)
    if len(devices) == 1:
        return next(iter(devices))
    return ", ".join(sorted(devices))


def _device_summary_from_observed(observed_devices: set[str], requested_device: torch.device) -> str:
    if not observed_devices:
        return str(requested_device)
    if len(observed_devices) == 1:
        return next(iter(observed_devices))
    return ", ".join(sorted(observed_devices))


def infer_unet_prefix_from_keys(keys: list[str] | tuple[str, ...]) -> str:
    counts = {candidate: 0 for candidate in _UNET_PREFIX_CANDIDATES}
    for key in keys:
        for candidate in _UNET_PREFIX_CANDIDATES:
            if key.startswith(candidate):
                counts[candidate] += 1
                break
    top = max(counts, key=counts.get)
    return top if counts[top] > 5 else "model."


def load_safetensors_state_dict(
    ckpt: str,
    requested_device: torch.device,
    *,
    return_metadata: bool = False,
    selected_keys: dict[str, str] | None = None,
) -> tuple[dict[str, Any], Any, str, str]:
    import comfy.memory_management
    import comfy.utils as comfy_utils

    metadata = None
    if comfy.memory_management.aimdo_enabled and requested_device.type == "cpu" and selected_keys is None:
        sd, metadata = comfy_utils.load_safetensors(ckpt)
        if not return_metadata:
            metadata = None
        return sd, metadata, "cpu", "aimdo_cpu"

    disable_mmap = getattr(comfy_utils, "DISABLE_MMAP", False)

    def read_handle(device_arg: Any, *, move_to_requested_device: bool) -> tuple[dict[str, Any], Any, str]:
        observed_devices: set[str] = set()
        with safe_open(ckpt, framework="pt", device=device_arg) as handle:
            key_map = selected_keys if selected_keys is not None else {key: key for key in handle.keys()}
            sd: dict[str, Any] = {}
            for source_key, target_key in key_map.items():
                tensor = handle.get_tensor(source_key)
                loaded = _prepare_loaded_tensor(
                    source_key,
                    tensor,
                    requested_device,
                    disable_mmap=disable_mmap,
                    move_to_requested_device=move_to_requested_device,
                )
                sd[target_key] = loaded
                if torch.is_tensor(loaded):
                    observed_devices.add(str(loaded.device))
            handle_metadata = handle.metadata() if return_metadata else None
        return sd, handle_metadata, _device_summary_from_observed(observed_devices, requested_device)

    try:
        safe_device = _safe_open_device_arg(requested_device)
        sd, metadata, actual_device = read_handle(safe_device, move_to_requested_device=False)
        return sd, metadata, actual_device, "direct"
    except Exception as exc:
        if requested_device.type == "cuda":
            _LOG.warning(
                "GPU Resident Loader: direct GPU safetensors load failed for %s; falling back to CPU path: %s",
                ckpt,
                exc,
            )
            try:
                sd, metadata, actual_device = read_handle("cpu", move_to_requested_device=True)
                return sd, metadata, actual_device, "cpu_then_copy"
            except Exception as fallback_exc:
                raise fallback_exc from exc
        raise


def _warn_pickle_gpu_compatibility(path: str, requested_device: torch.device) -> None:
    if requested_device.type != "cuda" or path in _WARNED_PICKLE_GPU_PATHS:
        return
    _WARNED_PICKLE_GPU_PATHS.add(path)
    _LOG.warning(
        "GPU Resident Loader: %s is not a safetensors file, so GPU-resident loading still goes through CPU-first torch.load(). "
        "Convert hot models with scripts/convert_checkpoint_to_safetensors.py for the narrow fast path.",
        path,
    )


def _sticky_protection_target(memory_required: int, device: Any) -> int:
    import comfy.model_management as model_management

    required = max(0, int(memory_required))
    target = required + adaptive_headroom_bytes(required)

    minimum_inference_memory = getattr(model_management, "minimum_inference_memory", None)
    if callable(minimum_inference_memory):
        try:
            target = max(target, int(minimum_inference_memory()))
        except Exception:
            pass

    get_total_memory = getattr(model_management, "get_total_memory", None)
    if callable(get_total_memory):
        try:
            total_memory = int(get_total_memory(device))
        except Exception:
            total_memory = 0
        if total_memory > 0:
            target = max(
                target,
                min(
                    _STICKY_PROTECTION_VRAM_FLOOR_CEIL_BYTES,
                    int(total_memory * _STICKY_PROTECTION_VRAM_FLOOR_RATIO),
                ),
            )

    return target


def _resolved_context(kind: str, source_path: str | None) -> tuple[torch.device | None, str, str | None]:
    ctx = REGISTRY.current_context()
    if ctx is not None and ctx.explicit_device is not None:
        return ctx.explicit_device, ctx.kind, ctx.note
    return REGISTRY.explicit_load_device(kind=kind, source_path=source_path), kind, None


def _record_generic_load(
    *,
    path: str,
    method: str,
    requested_device: torch.device | None,
    actual_device: str,
    note: str | None = None,
    error: str | None = None,
) -> None:
    ctx = REGISTRY.current_context()
    kind = ctx.kind if ctx is not None else "unknown"
    REGISTRY.record_load(
        path=path,
        kind=kind,
        method=method,
        requested_device=_device_string(requested_device),
        actual_device=actual_device,
        note=note or (ctx.note if ctx is not None else None),
        error=error,
    )


def _patched_load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
    import comfy.memory_management
    import comfy.utils as comfy_utils

    requested_device = _normalize_device(device)
    ctx = REGISTRY.current_context()
    if requested_device is None and ctx is not None and ctx.explicit_device is not None:
        requested_device = ctx.explicit_device
    if requested_device is None:
        requested_device = torch.device("cpu")

    metadata = None
    lowered = str(ckpt).lower()

    if lowered.endswith((".safetensors", ".sft")):
        try:
            selected_keys = None
            selected_suffix = None
            if ctx is not None and ctx.kind in {KIND_MODEL, KIND_CLIP, KIND_VAE}:
                selected_keys = _selected_component_keys_from_header(ckpt, ctx.kind)
                selected_suffix = _selected_component_suffix(ctx.kind)

            sd, metadata, actual_device, load_mode = load_safetensors_state_dict(
                ckpt,
                requested_device,
                return_metadata=return_metadata,
                selected_keys=selected_keys,
            )
            if load_mode == "aimdo_cpu":
                method = "safetensors_aimdo_cpu"
            elif selected_keys is not None and selected_suffix is not None:
                method = f"safetensors_cpu_then_copy_to_cuda_{selected_suffix}" if load_mode == "cpu_then_copy" else (
                    f"safetensors_gpu_direct_{selected_suffix}" if requested_device.type == "cuda" else f"safetensors_cpu_{selected_suffix}"
                )
            else:
                method = "safetensors_cpu_then_copy_to_cuda" if load_mode == "cpu_then_copy" else (
                    "safetensors_gpu_direct" if requested_device.type == "cuda" else "safetensors_cpu"
                )
            _record_generic_load(
                path=ckpt,
                method=method,
                requested_device=requested_device,
                actual_device=actual_device,
            )
            return (sd, metadata) if return_metadata else sd
        except Exception as exc:
            if len(getattr(exc, "args", ())) > 0:
                message = exc.args[0]
                if isinstance(message, str):
                    if "HeaderTooLarge" in message:
                        raise ValueError(
                            f"{message}\n\nFile path: {ckpt}\n\n"
                            "The safetensors file is corrupt or invalid. Make sure this is actually a "
                            "safetensors file and not a ckpt or pt or other filetype."
                        ) from exc
                    if "MetadataIncompleteBuffer" in message:
                        raise ValueError(
                            f"{message}\n\nFile path: {ckpt}\n\n"
                            "The safetensors file is corrupt/incomplete. Check the file size and make sure "
                            "you have copied/downloaded it correctly."
                        ) from exc
            _record_generic_load(
                path=ckpt,
                method="safetensors_load_failed",
                requested_device=requested_device,
                actual_device="error",
                error=str(exc),
            )
            raise

    torch_args = {}
    if getattr(comfy_utils, "MMAP_TORCH_FILES", False):
        torch_args["mmap"] = True

    _warn_pickle_gpu_compatibility(ckpt, requested_device)
    torch_load_device = torch.device("cpu") if requested_device.type == "cuda" else requested_device
    pl_sd = torch.load(ckpt, map_location=torch_load_device, weights_only=True, **torch_args)
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        if len(pl_sd) == 1:
            key = list(pl_sd.keys())[0]
            sd = pl_sd[key]
            if not isinstance(sd, dict):
                sd = pl_sd
        else:
            sd = pl_sd

    if isinstance(sd, dict):
        for key, value in list(sd.items()):
            if torch.is_tensor(value):
                sd[key] = _prepare_loaded_tensor(
                    key,
                    value,
                    requested_device,
                    disable_mmap=False,
                    move_to_requested_device=True,
                )

    method = "torch_load_cpu_first_to_cuda" if requested_device.type == "cuda" else "torch_load_cpu"
    _record_generic_load(
        path=ckpt,
        method=method,
        requested_device=requested_device,
        actual_device=_state_dict_device_summary(sd, requested_device) if isinstance(sd, dict) else str(requested_device),
    )
    return (sd, metadata) if return_metadata else sd


def _bind_checkpoint_outputs(result, source_path: str) -> None:
    if not result:
        return
    model = result[0] if len(result) > 0 else None
    clip = result[1] if len(result) > 1 else None
    vae = result[2] if len(result) > 2 else None
    if model is not None:
        REGISTRY.bind_object(model, source_path=source_path, kind=KIND_MODEL, note="checkpoint model")
    if clip is not None and getattr(clip, "patcher", None) is not None:
        REGISTRY.bind_object(clip.patcher, source_path=source_path, kind=KIND_CLIP, note="checkpoint clip")
    if vae is not None and getattr(vae, "patcher", None) is not None:
        REGISTRY.bind_object(vae.patcher, source_path=source_path, kind=KIND_VAE, note="checkpoint vae")




def _bind_diffusers_outputs(result, source_path: str) -> None:
    if not result:
        return
    model = result[0] if len(result) > 0 else None
    clip = result[1] if len(result) > 1 else None
    vae = result[2] if len(result) > 2 else None
    if model is not None:
        REGISTRY.bind_object(model, source_path=source_path, kind=KIND_MODEL, note="diffusers model")
    if clip is not None and getattr(clip, "patcher", None) is not None:
        REGISTRY.bind_object(clip.patcher, source_path=source_path, kind=KIND_CLIP, note="diffusers clip")
    if vae is not None and getattr(vae, "patcher", None) is not None:
        REGISTRY.bind_object(vae.patcher, source_path=source_path, kind=KIND_VAE, note="diffusers vae")


def _wrap_with_load_context(kind: str, path_arg_index: int = 0, bind_output: Callable[[Any, str], None] | None = None):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            source_path = None
            if len(args) > path_arg_index:
                source_path = args[path_arg_index]
            explicit_device = REGISTRY.explicit_load_device(kind=kind, source_path=source_path)
            with REGISTRY.load_context(kind=kind, source_path=source_path, explicit_device=explicit_device):
                result = func(*args, **kwargs)
            if bind_output is not None and source_path is not None:
                bind_output(result, source_path)
            return result

        return wrapper

    return decorator


def _wrap_load_clip(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ckpt_paths = args[0] if args else kwargs.get("ckpt_paths")
        source_path = None
        if isinstance(ckpt_paths, (list, tuple)) and ckpt_paths:
            source_path = ckpt_paths[0]
        explicit_device = REGISTRY.explicit_load_device(kind=KIND_CLIP, source_path=source_path)
        with REGISTRY.load_context(kind=KIND_CLIP, source_path=source_path, explicit_device=explicit_device):
            clip = func(*args, **kwargs)
        if clip is not None and getattr(clip, "patcher", None) is not None and source_path is not None:
            REGISTRY.bind_object(clip.patcher, source_path=source_path, kind=KIND_CLIP)
        return clip

    return wrapper


def _sticky_safe_batch_number(*, batch_count: int, free_memory: int, memory_used: int, device: Any) -> int:
    batches = max(1, int(max(0, int(free_memory)) / max(1, int(memory_used))))
    batches = min(max(1, int(batch_count)), batches)
    if REGISTRY.get_policy() != "sticky_gpu" or device is None:
        return batches

    reserve = max(0, _sticky_protection_target(memory_used, device) - max(0, int(memory_used)))
    safe_budget = max(0, int(free_memory) - reserve)
    safe_batches = max(1, int(safe_budget / max(1, int(memory_used))))
    capped = min(batches, safe_batches)
    if capped < batches:
        _LOG.debug(
            "GPU Resident Loader: capped VAE batch from %s to %s to preserve %s bytes of transient headroom.",
            batches,
            capped,
            reserve,
        )
    return max(1, capped)


def _scaled_batch_memory(total_memory_used: int, total_batch_count: int, batch_number: int) -> int:
    total_memory = max(1, int(total_memory_used))
    total_batches = max(1, int(total_batch_count))
    current_batch = max(1, min(int(batch_number), total_batches))
    return max(1, (total_memory * current_batch + total_batches - 1) // total_batches)


def _sticky_vae_free_memory(*, device: Any, patcher: Any) -> int:
    import comfy.model_management as model_management

    get_free_memory = getattr(model_management, "get_free_memory", None)
    if callable(get_free_memory):
        try:
            return max(0, int(get_free_memory(device)))
        except Exception:
            pass

    return max(0, int(patcher.get_free_memory(device)))


def _prepare_sticky_vae_batch(
    *,
    device: Any,
    patcher: Any,
    total_memory_used: int,
    total_batch_count: int,
) -> tuple[int, int, bool]:
    free_memory = _sticky_vae_free_memory(device=device, patcher=patcher)
    batch_number = _sticky_safe_batch_number(
        batch_count=total_batch_count,
        free_memory=free_memory,
        memory_used=total_memory_used,
        device=device,
    )
    batch_memory_used = _scaled_batch_memory(total_memory_used, total_batch_count, batch_number)

    if REGISTRY.get_policy() != "sticky_gpu" or device is None:
        return batch_number, batch_memory_used, False

    target_free = _sticky_protection_target(batch_memory_used, device)
    if free_memory < target_free:
        try:
            trim_resident_vram(
                device=device,
                target_free_vram_bytes=target_free,
                respect_sticky=True,
                sticky_floor_priority=0,
                allow_partial_unload=True,
                keep_models=(patcher,),
            )
        except Exception as exc:
            _LOG.debug("GPU Resident Loader: proactive VAE trim failed for %s bytes: %s", batch_memory_used, exc)

        free_memory = _sticky_vae_free_memory(device=device, patcher=patcher)
        batch_number = _sticky_safe_batch_number(
            batch_count=total_batch_count,
            free_memory=free_memory,
            memory_used=total_memory_used,
            device=device,
        )
        batch_memory_used = _scaled_batch_memory(total_memory_used, total_batch_count, batch_number)
        target_free = _sticky_protection_target(batch_memory_used, device)

    should_tile = free_memory < target_free and batch_number <= 1
    if should_tile:
        _LOG.info(
            "GPU Resident Loader: skipping regular VAE pass and switching directly to tiled mode; free=%s target=%s batch_memory=%s",
            free_memory,
            target_free,
            batch_memory_used,
        )
    return batch_number, batch_memory_used, should_tile


def _wrap_vae_encode(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(self, pixel_samples):
        if REGISTRY.get_policy() != "sticky_gpu":
            return func(self, pixel_samples)

        import comfy.model_management as model_management

        self.throw_exception_if_invalid()
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        pixel_samples = pixel_samples.movedim(-1, 1)
        do_tile = False
        if self.latent_dim == 3 and pixel_samples.ndim < 5:
            if not self.not_video:
                pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)
            else:
                pixel_samples = pixel_samples.unsqueeze(2)
        try:
            memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
            batch_number, batch_memory_used, should_tile = _prepare_sticky_vae_batch(
                device=self.device,
                patcher=self.patcher,
                total_memory_used=memory_used,
                total_batch_count=pixel_samples.shape[0],
            )
            if should_tile:
                do_tile = True
            else:
                model_management.load_models_gpu(
                    [self.patcher],
                    memory_required=batch_memory_used,
                    force_full_load=self.disable_offload,
                )
                samples = None
                for x in range(0, pixel_samples.shape[0], batch_number):
                    pixels_in = self.process_input(pixel_samples[x:x + batch_number]).to(self.vae_dtype)
                    if getattr(self.first_stage_model, "comfy_has_chunked_io", False):
                        out = self.first_stage_model.encode(pixels_in, device=self.device)
                    else:
                        pixels_in = pixels_in.to(self.device)
                        out = self.first_stage_model.encode(pixels_in)
                    out = out.to(self.output_device).to(dtype=self.vae_output_dtype())
                    if samples is None:
                        samples = torch.empty(
                            (pixel_samples.shape[0],) + tuple(out.shape[1:]),
                            device=self.output_device,
                            dtype=self.vae_output_dtype(),
                        )
                    samples[x:x + batch_number] = out
        except Exception as e:
            model_management.raise_non_oom(e)
            _LOG.warning("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
            do_tile = True

        if do_tile:
            model_management.soft_empty_cache()
            if self.latent_dim == 3:
                tile = 256
                overlap = tile // 4
                samples = self.encode_tiled_3d(pixel_samples, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap))
            elif self.latent_dim == 1 or self.extra_1d_channel is not None:
                samples = self.encode_tiled_1d(pixel_samples)
            else:
                samples = self.encode_tiled_(pixel_samples)

        return samples

    return wrapper


def _default_tiled_vae_axes(
    *,
    latent_dim: int,
    extra_1d_channel: Any,
    tile_x: int | None,
    tile_y: int | None,
    tile_t: int | None,
    decode: bool,
) -> tuple[int | None, int | None, int | None]:
    if latent_dim == 3:
        default_tile_x = 32 if decode else 512
        default_tile_y = 32 if decode else 512
        default_tile_t = 999 if decode else 9999
    elif latent_dim == 1 or extra_1d_channel is not None:
        default_tile_x = 256 * 2048
        default_tile_y = None
        default_tile_t = None
    else:
        default_tile_x = 64 if decode else 512
        default_tile_y = 64 if decode else 512
        default_tile_t = None

    resolved_tile_x = default_tile_x if tile_x is None else max(1, int(tile_x))
    resolved_tile_y = default_tile_y if tile_y is None else max(1, int(tile_y))
    resolved_tile_t = default_tile_t if tile_t is None else max(1, int(tile_t))
    return resolved_tile_x, resolved_tile_y, resolved_tile_t


def _shape_with_capped_tail(shape: tuple[int, ...], tail_caps: dict[int, int | None]) -> tuple[int, ...]:
    capped = list(shape)
    for index, cap in tail_caps.items():
        if cap is None:
            continue
        capped[index] = min(int(capped[index]), max(1, int(cap)))
    return tuple(capped)


def _tiled_vae_memory_shapes(
    *,
    shape: tuple[int, ...],
    latent_dim: int,
    extra_1d_channel: Any,
    tile_x: int | None,
    tile_y: int | None,
    tile_t: int | None,
    decode: bool,
) -> list[tuple[int, ...]]:
    resolved_tile_x, resolved_tile_y, resolved_tile_t = _default_tiled_vae_axes(
        latent_dim=latent_dim,
        extra_1d_channel=extra_1d_channel,
        tile_x=tile_x,
        tile_y=tile_y,
        tile_t=tile_t,
        decode=decode,
    )

    if latent_dim == 3:
        return [
            _shape_with_capped_tail(
                shape,
                {
                    len(shape) - 3: resolved_tile_t,
                    len(shape) - 2: resolved_tile_y,
                    len(shape) - 1: resolved_tile_x,
                },
            )
        ]

    if latent_dim == 1 or extra_1d_channel is not None:
        return [_shape_with_capped_tail(shape, {len(shape) - 1: resolved_tile_x})]

    if decode:
        return [
            _shape_with_capped_tail(
                shape,
                {
                    len(shape) - 2: resolved_tile_y,
                    len(shape) - 1: resolved_tile_x,
                },
            )
        ]

    return [
        _shape_with_capped_tail(
            shape,
            {
                len(shape) - 2: resolved_tile_y,
                len(shape) - 1: resolved_tile_x,
            },
        ),
        _shape_with_capped_tail(
            shape,
            {
                len(shape) - 2: max(1, resolved_tile_y // 2),
                len(shape) - 1: max(1, resolved_tile_x * 2),
            },
        ),
        _shape_with_capped_tail(
            shape,
            {
                len(shape) - 2: max(1, resolved_tile_y * 2),
                len(shape) - 1: max(1, resolved_tile_x // 2),
            },
        ),
    ]


@contextlib.contextmanager
def _temporary_tiled_vae_memory_estimate(
    self,
    *,
    decode: bool,
    tile_x: int | None,
    tile_y: int | None,
    tile_t: int | None,
) -> Any:
    memory_attr = "memory_used_decode" if decode else "memory_used_encode"
    original = getattr(self, memory_attr, None)
    if not callable(original):
        yield
        return
    had_instance_attr = memory_attr in getattr(self, "__dict__", {})
    lock = getattr(self, _TILED_VAE_MEMORY_LOCK_ATTR, None)
    if lock is None:
        with _TILED_VAE_LOCK_INIT:
            lock = getattr(self, _TILED_VAE_MEMORY_LOCK_ATTR, None)
            if lock is None:
                lock = threading.RLock()
                setattr(self, _TILED_VAE_MEMORY_LOCK_ATTR, lock)

    def estimated(shape, dtype, *args, **kwargs):
        shapes = _tiled_vae_memory_shapes(
            shape=tuple(int(dim) for dim in shape),
            latent_dim=int(getattr(self, "latent_dim", 2)),
            extra_1d_channel=getattr(self, "extra_1d_channel", None),
            tile_x=tile_x,
            tile_y=tile_y,
            tile_t=tile_t,
            decode=decode,
        )
        return max(int(original(candidate, dtype, *args, **kwargs)) for candidate in shapes)

    with lock:
        setattr(self, memory_attr, estimated)
        try:
            yield
        finally:
            if had_instance_attr:
                setattr(self, memory_attr, original)
            else:
                delattr(self, memory_attr)


@functools.lru_cache(maxsize=None)
def _tiled_vae_supported_kwargs(func: Callable[..., Any]) -> frozenset[str]:
    return frozenset(inspect.signature(func).parameters)


def _call_tiled_vae(
    func: Callable[..., Any],
    self,
    data,
    *,
    tile_x=None,
    tile_y=None,
    overlap=None,
    tile_t=None,
    overlap_t=None,
):
    kwargs = {}
    supported_kwargs = _tiled_vae_supported_kwargs(func)
    if "tile_x" in supported_kwargs:
        kwargs["tile_x"] = tile_x
    if "tile_y" in supported_kwargs:
        kwargs["tile_y"] = tile_y
    if "overlap" in supported_kwargs:
        kwargs["overlap"] = overlap
    if "tile_t" in supported_kwargs:
        kwargs["tile_t"] = tile_t
    if "overlap_t" in supported_kwargs:
        kwargs["overlap_t"] = overlap_t
    return func(self, data, **kwargs)


def _should_prefer_tiled_vae_encode(vae: Any, pixel_samples: Any) -> bool:
    if REGISTRY.get_policy() != "sticky_gpu":
        return False
    if vae is None or pixel_samples is None:
        return False

    try:
        vae.throw_exception_if_invalid()
        prepared = vae.vae_encode_crop_pixels(pixel_samples)
        prepared = prepared.movedim(-1, 1)
        if int(getattr(vae, "latent_dim", 2)) == 3 and prepared.ndim < 5:
            if not getattr(vae, "not_video", False):
                prepared = prepared.movedim(1, 0).unsqueeze(0)
            else:
                prepared = prepared.unsqueeze(2)

        memory_used = vae.memory_used_encode(prepared.shape, vae.vae_dtype)
        _, _, should_tile = _prepare_sticky_vae_batch(
            device=getattr(vae, "device", None),
            patcher=getattr(vae, "patcher", None),
            total_memory_used=memory_used,
            total_batch_count=prepared.shape[0],
        )
        return bool(should_tile)
    except Exception as exc:
        _LOG.debug("GPU Resident Loader: failed to preflight sticky VAE encode preference: %s", exc)
        return False


def _call_bound_tiled_vae(func: Callable[..., Any], pixel_samples: Any, *args: Any, **kwargs: Any) -> Any:
    supported_kwargs = _tiled_vae_supported_kwargs(getattr(func, "__func__", func))
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_kwargs}
    return func(pixel_samples, *args, **filtered_kwargs)


@contextlib.contextmanager
def _temporary_prefer_tiled_vae_encode(vae: Any):
    original_encode = getattr(vae, "encode", None)
    encode_tiled = getattr(vae, "encode_tiled", None)
    if not callable(original_encode) or not callable(encode_tiled):
        yield
        return

    had_instance_attr = "encode" in getattr(vae, "__dict__", {})
    lock = getattr(vae, _TILED_VAE_MEMORY_LOCK_ATTR, None)
    if lock is None:
        with _TILED_VAE_LOCK_INIT:
            lock = getattr(vae, _TILED_VAE_MEMORY_LOCK_ATTR, None)
            if lock is None:
                lock = threading.RLock()
                setattr(vae, _TILED_VAE_MEMORY_LOCK_ATTR, lock)

    def prefer_encode(pixel_samples, *args, **kwargs):
        if _should_prefer_tiled_vae_encode(vae, pixel_samples):
            return _call_bound_tiled_vae(encode_tiled, pixel_samples, *args, **kwargs)
        return original_encode(pixel_samples, *args, **kwargs)

    with lock:
        setattr(vae, "encode", prefer_encode)
        try:
            yield
        finally:
            if had_instance_attr:
                setattr(vae, "encode", original_encode)
            else:
                delattr(vae, "encode")


def _wrap_vae_encode_for_inpaint_node(func: Callable[..., Any]) -> Callable[..., Any]:
    supported_kwargs = frozenset(inspect.signature(func).parameters)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Filter kwargs to only include those supported by the wrapped function
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_kwargs}

        # Supply default grow_mask_by if not present and supported
        if "grow_mask_by" in supported_kwargs and "grow_mask_by" not in filtered_kwargs:
            filtered_kwargs["grow_mask_by"] = 6

        if REGISTRY.get_policy() != "sticky_gpu":
            return func(self, *args, **filtered_kwargs)

        # Extract vae from args for the context manager
        vae = args[0] if args else kwargs.get("vae")
        with _temporary_prefer_tiled_vae_encode(vae):
            return func(self, *args, **filtered_kwargs)

    return wrapper


def _wrap_inpaint_model_conditioning_node(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(self, positive, negative, pixels, vae, mask, noise_mask=True):
        if REGISTRY.get_policy() != "sticky_gpu":
            return func(self, positive, negative, pixels, vae, mask, noise_mask=noise_mask)
        with _temporary_prefer_tiled_vae_encode(vae):
            return func(self, positive, negative, pixels, vae, mask, noise_mask=noise_mask)

    return wrapper


def _wrap_vae_encode_tiled(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(self, pixel_samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        if REGISTRY.get_policy() != "sticky_gpu":
            return _call_tiled_vae(
                func,
                self,
                pixel_samples,
                tile_x=tile_x,
                tile_y=tile_y,
                overlap=overlap,
                tile_t=tile_t,
                overlap_t=overlap_t,
            )

        with _temporary_tiled_vae_memory_estimate(
            self,
            decode=False,
            tile_x=tile_x,
            tile_y=tile_y,
            tile_t=tile_t,
        ):
            return _call_tiled_vae(
                func,
                self,
                pixel_samples,
                tile_x=tile_x,
                tile_y=tile_y,
                overlap=overlap,
                tile_t=tile_t,
                overlap_t=overlap_t,
            )

    return wrapper


def _wrap_vae_decode_tiled(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        if REGISTRY.get_policy() != "sticky_gpu":
            return _call_tiled_vae(
                func,
                self,
                samples,
                tile_x=tile_x,
                tile_y=tile_y,
                overlap=overlap,
                tile_t=tile_t,
                overlap_t=overlap_t,
            )

        with _temporary_tiled_vae_memory_estimate(
            self,
            decode=True,
            tile_x=tile_x,
            tile_y=tile_y,
            tile_t=tile_t,
        ):
            return _call_tiled_vae(
                func,
                self,
                samples,
                tile_x=tile_x,
                tile_y=tile_y,
                overlap=overlap,
                tile_t=tile_t,
                overlap_t=overlap_t,
            )

    return wrapper


def _wrap_vae_decode(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(self, samples_in, vae_options={}):
        if REGISTRY.get_policy() != "sticky_gpu":
            return func(self, samples_in, vae_options)

        import comfy.model_management as model_management

        self.throw_exception_if_invalid()
        pixel_samples = None
        do_tile = False
        if self.latent_dim == 2 and samples_in.ndim == 5:
            samples_in = samples_in[:, :, 0]
        try:
            memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
            batch_number, batch_memory_used, should_tile = _prepare_sticky_vae_batch(
                device=self.device,
                patcher=self.patcher,
                total_memory_used=memory_used,
                total_batch_count=samples_in.shape[0],
            )

            preallocated = False
            if should_tile:
                do_tile = True
            else:
                model_management.load_models_gpu(
                    [self.patcher],
                    memory_required=batch_memory_used,
                    force_full_load=self.disable_offload,
                )
                if getattr(self.first_stage_model, "comfy_has_chunked_io", False):
                    pixel_samples = torch.empty(
                        self.first_stage_model.decode_output_shape(samples_in.shape),
                        device=self.output_device,
                        dtype=self.vae_output_dtype(),
                    )
                    preallocated = True
                for x in range(0, samples_in.shape[0], batch_number):
                    samples = samples_in[x:x + batch_number].to(device=self.device, dtype=self.vae_dtype)
                    if preallocated:
                        self.first_stage_model.decode(samples, output_buffer=pixel_samples[x:x + batch_number], **vae_options)
                    else:
                        out = self.first_stage_model.decode(samples, **vae_options).to(
                            device=self.output_device,
                            dtype=self.vae_output_dtype(),
                            copy=True,
                        )
                        if pixel_samples is None:
                            pixel_samples = torch.empty(
                                (samples_in.shape[0],) + tuple(out.shape[1:]),
                                device=self.output_device,
                                dtype=self.vae_output_dtype(),
                            )
                        pixel_samples[x:x + batch_number].copy_(out)
                        del out
                    self.process_output(pixel_samples[x:x + batch_number])
        except Exception as e:
            model_management.raise_non_oom(e)
            _LOG.warning("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            do_tile = True

        if do_tile:
            model_management.soft_empty_cache()
            dims = samples_in.ndim - 2
            if dims == 1 or self.extra_1d_channel is not None:
                pixel_samples = self.decode_tiled_1d(samples_in)
            elif dims == 2:
                pixel_samples = self.decode_tiled_(samples_in)
            elif dims == 3:
                tile = 256 // self.spacial_compression_decode()
                overlap = tile // 4
                pixel_samples = self.decode_tiled_3d(samples_in, tile_x=tile, tile_y=tile, overlap=(1, overlap, overlap))

        pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
        return pixel_samples

    return wrapper


def _wrap_load_models_gpu(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(models, *args, **kwargs):
        import comfy.model_management as model_management

        requested_models = set()
        for model in list(models):
            requested_models.add(model)
            for additional in model.model_patches_models():
                requested_models.add(additional)

        clone_conflicts: list[Any] = []
        seen_loaded_ids: set[int] = set()
        for requested in requested_models:
            is_clone = getattr(requested, "is_clone", None)
            if not callable(is_clone):
                continue
            for loaded in list(model_management.current_loaded_models):
                try:
                    dead = loaded.is_dead()
                except Exception:
                    dead = False
                if id(loaded) in seen_loaded_ids or dead:
                    continue
                loaded_model = getattr(loaded, "model", None)
                if loaded_model is None or loaded_model is requested:
                    continue
                try:
                    if not requested.is_clone(loaded_model):
                        continue
                except Exception as exc:
                    raise RuntimeError(
                        "GPU Resident Loader: failed to evaluate clone-conflict state before replacement"
                    ) from exc
                clone_conflicts.append(loaded)
                seen_loaded_ids.add(id(loaded))

        clone_conflicts_unloaded = 0
        try:
            for loaded in clone_conflicts:
                # ComfyUI's built-in clone replacement pops the wrapper and only calls detach(False),
                # which does not unpatch base weights. Fully unload before replacement or fail closed.
                if not unload_loaded_model(
                    loaded,
                    active_device=getattr(loaded, "device", None),
                    force_offload_to_cpu=True,
                ):
                    raise RuntimeError("GPU Resident Loader: failed to fully unload a clone-conflict wrapper before replacement")
                try:
                    model_management.current_loaded_models.remove(loaded)
                except ValueError:
                    pass
                clone_conflicts_unloaded += 1
        finally:
            if clone_conflicts_unloaded > 0:
                if hasattr(model_management, "soft_empty_cache"):
                    model_management.soft_empty_cache()
                REGISTRY.refresh_runtime_state()

        result = func(models, *args, **kwargs)
        for model in list(models):
            REGISTRY.touch(model)
        REGISTRY.refresh_runtime_state()
        return result

    return wrapper


def _wrap_loaded_model_unload(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(self, memory_to_free=None, unpatch_weights=True):
        model = getattr(self, "model", None)
        loaded_device = getattr(self, "device", None)
        if not _cpu_offload_required(model, loaded_device):
            return func(self, memory_to_free=memory_to_free, unpatch_weights=unpatch_weights)

        with _temporary_offload_device(model, torch.device("cpu")) as redirected:
            if redirected:
                _LOG.debug(
                    "GPU Resident Loader: redirecting unload of %s from %s to CPU to reclaim VRAM",
                    type(getattr(model, "model", model)).__name__,
                    _device_string(_normalize_device(loaded_device)),
                )
            return func(self, memory_to_free=memory_to_free, unpatch_weights=unpatch_weights)

    return wrapper


def _wrap_model_patcher_detach(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(self, unpatch_all=True):
        if not _cpu_offload_required(self, getattr(self.model, "device", None)):
            return func(self, unpatch_all=unpatch_all)

        with _temporary_offload_device(self, torch.device("cpu")) as redirected:
            if redirected:
                _LOG.debug(
                    "GPU Resident Loader: redirecting detach of %s from %s to CPU to reclaim VRAM",
                    type(getattr(self, "model", self)).__name__,
                    _device_string(_normalize_device(getattr(self.model, "device", None))),
                )
            return func(self, unpatch_all=unpatch_all)

    return wrapper


def _wrap_free_memory(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wraps a free-memory function to enforce sticky-GPU protection and external fallback trimming.
    
    When the registry policy is "sticky_gpu" and a device is provided, the wrapper:
    - Reserves VRAM for sticky-loaded models by attempting a pre-trim to a computed protection target.
    - Protects a subset of sticky-loaded wrappers from unloading when calling the original function by adding them to `keep_loaded`.
    - After the original free-memory call, refreshes both REGISTRY and EXTERNAL_REGISTRY runtime state.
    - If external trimming is available and still needed, attempts a fallback trim that includes external residency.
      Dynamic free-memory calls are excluded because Comfy reduces their effective target internally.
    
    Parameters:
        memory_required: Number of bytes the caller needs to free.
        device: Target device for which memory is being freed (may be None).
        keep_loaded: Iterable of loaded-wrapper objects that must be kept; the wrapper may extend this list with additional protected wrappers.
    
    Returns:
        The value returned by the wrapped `func`.
        
    Notes:
    - The wrapper may call `trim_resident_vram` and `model_management.get_free_memory`; exceptions from trimming or free-memory queries are caught and logged, not propagated.
    - Side effects include invoking trims and refreshing runtime state on REGISTRY and EXTERNAL_REGISTRY.
    """
    @functools.wraps(func)
    def wrapper(memory_required, device, keep_loaded=None, *args, **kwargs):
        import comfy.model_management as model_management

        keep_loaded = list(keep_loaded or [])
        protected_wrappers: list[Any] = []
        sticky_wrappers: list[Any] = []
        if REGISTRY.get_policy() == "sticky_gpu" and device is not None:
            sticky_wrappers = [w for w in REGISTRY.sticky_loaded_wrappers(device) if w not in keep_loaded]
        if device is not None and sticky_wrappers:
            protection_target = _sticky_protection_target(memory_required, device)
            keep_models = tuple(
                model
                for model in (getattr(loaded_wrapper, "model", None) for loaded_wrapper in keep_loaded)
                if model is not None
            )
            try:
                trim_resident_vram(
                    device=device,
                    target_free_vram_bytes=protection_target,
                    respect_sticky=True,
                    sticky_floor_priority=0,
                    allow_partial_unload=True,
                    keep_models=keep_models,
                )
            except Exception as exc:
                _LOG.debug("GPU Resident Loader: sticky pre-trim failed for free_memory(%s): %s", memory_required, exc)

            sticky_wrappers = [w for w in REGISTRY.sticky_loaded_wrappers(device) if w not in keep_loaded]
            try:
                free_now = model_management.get_free_memory(device)
            except Exception:
                free_now = None
            if free_now is None:
                protected_wrappers = sticky_wrappers
            else:
                unloadable_wrappers = []
                for loaded in list(model_management.current_loaded_models):
                    if loaded.device == device and loaded not in keep_loaded and not loaded.is_dead():
                        unloadable_wrappers.append(loaded)
                available_for_protection = max(
                    0,
                    free_now + sum(max(0, loaded.model_loaded_memory()) for loaded in unloadable_wrappers) - protection_target,
                )
                protected_memory = 0
                for loaded in sticky_wrappers:
                    estimated_memory = max(0, loaded.model_loaded_memory())
                    if protected_memory + estimated_memory <= available_for_protection:
                        protected_wrappers.append(loaded)
                        protected_memory += estimated_memory

        unloaded = func(memory_required, device, keep_loaded + protected_wrappers, *args, **kwargs)

        REGISTRY.refresh_runtime_state()
        EXTERNAL_REGISTRY.refresh_runtime_state()

        for_dynamic = bool(kwargs.get("for_dynamic", args[0] if args else False))
        if device is not None and not external_trim_enabled() and not for_dynamic:
            fallback_target = memory_required
            if REGISTRY.get_policy() == "sticky_gpu":
                fallback_target = max(fallback_target, _sticky_protection_target(memory_required, device))
            try:
                free_now = model_management.get_free_memory(device)
            except Exception:
                free_now = None
            if free_now is not None and int(free_now) < int(fallback_target):
                protected_models = tuple(
                    model
                    for model in (getattr(loaded_wrapper, "model", None) for loaded_wrapper in keep_loaded + protected_wrappers)
                    if model is not None
                )
                keep_models = protected_models + external_objects_for_models(protected_models)
                try:
                    trim_resident_vram(
                        device=device,
                        target_free_vram_bytes=int(fallback_target),
                        respect_sticky=True,
                        sticky_floor_priority=0,
                        allow_partial_unload=True,
                        keep_models=keep_models,
                        include_external=True,
                    )
                except Exception as exc:
                    _LOG.debug(
                        "GPU Resident Loader: external fallback trim failed for free_memory(%s): %s",
                        memory_required,
                        exc,
                    )
                REGISTRY.refresh_runtime_state()
                EXTERNAL_REGISTRY.refresh_runtime_state()
        return unloaded

    return wrapper


def _remember_original(key: str, value: Callable[..., Any]) -> Callable[..., Any]:
    return _ORIGINALS.setdefault(key, value)


def _patch_model_management_devices() -> None:
    import comfy.model_management as model_management

    kind_by_function = {
        "unet_offload_device": KIND_MODEL,
        "unet_inital_load_device": KIND_MODEL,
        "text_encoder_offload_device": KIND_CLIP,
        "text_encoder_device": KIND_CLIP,
        "vae_offload_device": KIND_VAE,
        "vae_device": KIND_VAE,
    }

    def wrap_device_func(name: str) -> None:
        key = f"model_management.{name}"
        original = _remember_original(key, getattr(model_management, name))
        if getattr(model_management, name) is not original:
            return

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            kind = kind_by_function.get(name)
            if not REGISTRY.wants_gpu_offload(kind):
                return result
            gpu_device = model_management.get_torch_device()
            if getattr(gpu_device, "type", None) == "cpu":
                return result
            return gpu_device

        setattr(model_management, name, wrapper)

    for name in (
        "unet_offload_device",
        "text_encoder_offload_device",
        "vae_offload_device",
        "text_encoder_device",
        "vae_device",
        "unet_inital_load_device",
    ):
        if hasattr(model_management, name):
            wrap_device_func(name)

    original_free_memory = _remember_original("model_management.free_memory", model_management.free_memory)
    if model_management.free_memory is original_free_memory:
        model_management.free_memory = _wrap_free_memory(original_free_memory)

    original_load_models_gpu = _remember_original("model_management.load_models_gpu", model_management.load_models_gpu)
    if model_management.load_models_gpu is original_load_models_gpu:
        model_management.load_models_gpu = _wrap_load_models_gpu(original_load_models_gpu)

    original_model_unload = _remember_original("model_management.LoadedModel.model_unload", model_management.LoadedModel.model_unload)
    if model_management.LoadedModel.model_unload is original_model_unload:
        model_management.LoadedModel.model_unload = _wrap_loaded_model_unload(original_model_unload)


def install_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import comfy.clip_vision as clip_vision
    import comfy.controlnet as controlnet
    import comfy.diffusers_load as diffusers_load
    import comfy.model_management as model_management
    import comfy.model_patcher as model_patcher
    import comfy.sd as comfy_sd
    import comfy.utils as comfy_utils
    import nodes as comfy_nodes

    original_load_torch_file = _remember_original("utils.load_torch_file", comfy_utils.load_torch_file)
    if comfy_utils.load_torch_file is original_load_torch_file:
        comfy_utils.load_torch_file = _patched_load_torch_file
    if hasattr(clip_vision, "load_torch_file"):
        original_clip_vision_load_torch_file = _remember_original("clip_vision.load_torch_file", clip_vision.load_torch_file)
        if clip_vision.load_torch_file is original_clip_vision_load_torch_file:
            clip_vision.load_torch_file = comfy_utils.load_torch_file

    _patch_model_management_devices()

    original_model_patcher_detach = _remember_original("model_patcher.ModelPatcher.detach", model_patcher.ModelPatcher.detach)
    if model_patcher.ModelPatcher.detach is original_model_patcher_detach:
        model_patcher.ModelPatcher.detach = _wrap_model_patcher_detach(original_model_patcher_detach)

    original_load_checkpoint_guess_config = _remember_original(
        "sd.load_checkpoint_guess_config",
        comfy_sd.load_checkpoint_guess_config,
    )
    if comfy_sd.load_checkpoint_guess_config is original_load_checkpoint_guess_config:
        comfy_sd.load_checkpoint_guess_config = _wrap_with_load_context(
            KIND_CHECKPOINT,
            path_arg_index=0,
            bind_output=_bind_checkpoint_outputs,
        )(original_load_checkpoint_guess_config)

    original_load_diffusion_model = _remember_original("sd.load_diffusion_model", comfy_sd.load_diffusion_model)
    if comfy_sd.load_diffusion_model is original_load_diffusion_model:
        comfy_sd.load_diffusion_model = _wrap_with_load_context(
            KIND_MODEL,
            path_arg_index=0,
            bind_output=lambda model, source_path: model is not None
            and REGISTRY.bind_object(model, source_path=source_path, kind=KIND_MODEL),
        )(original_load_diffusion_model)

    original_load_clip = _remember_original("sd.load_clip", comfy_sd.load_clip)
    if comfy_sd.load_clip is original_load_clip:
        comfy_sd.load_clip = _wrap_load_clip(original_load_clip)

    original_vae_encode = _remember_original("sd.VAE.encode", comfy_sd.VAE.encode)
    if comfy_sd.VAE.encode is original_vae_encode:
        comfy_sd.VAE.encode = _wrap_vae_encode(original_vae_encode)

    original_vae_decode = _remember_original("sd.VAE.decode", comfy_sd.VAE.decode)
    if comfy_sd.VAE.decode is original_vae_decode:
        comfy_sd.VAE.decode = _wrap_vae_decode(original_vae_decode)

    original_vae_encode_tiled = _remember_original("sd.VAE.encode_tiled", comfy_sd.VAE.encode_tiled)
    if comfy_sd.VAE.encode_tiled is original_vae_encode_tiled:
        comfy_sd.VAE.encode_tiled = _wrap_vae_encode_tiled(original_vae_encode_tiled)

    original_vae_decode_tiled = _remember_original("sd.VAE.decode_tiled", comfy_sd.VAE.decode_tiled)
    if comfy_sd.VAE.decode_tiled is original_vae_decode_tiled:
        comfy_sd.VAE.decode_tiled = _wrap_vae_decode_tiled(original_vae_decode_tiled)

    if hasattr(comfy_nodes, "VAEEncodeForInpaint") and hasattr(comfy_nodes.VAEEncodeForInpaint, "encode"):
        original_vae_encode_for_inpaint = _remember_original(
            "nodes.VAEEncodeForInpaint.encode",
            comfy_nodes.VAEEncodeForInpaint.encode,
        )
        if comfy_nodes.VAEEncodeForInpaint.encode is original_vae_encode_for_inpaint:
            comfy_nodes.VAEEncodeForInpaint.encode = _wrap_vae_encode_for_inpaint_node(original_vae_encode_for_inpaint)

    if hasattr(comfy_nodes, "InpaintModelConditioning") and hasattr(comfy_nodes.InpaintModelConditioning, "encode"):
        original_inpaint_model_conditioning = _remember_original(
            "nodes.InpaintModelConditioning.encode",
            comfy_nodes.InpaintModelConditioning.encode,
        )
        if comfy_nodes.InpaintModelConditioning.encode is original_inpaint_model_conditioning:
            comfy_nodes.InpaintModelConditioning.encode = _wrap_inpaint_model_conditioning_node(original_inpaint_model_conditioning)

    original_clip_vision_load = _remember_original("clip_vision.load", clip_vision.load)
    if clip_vision.load is original_clip_vision_load:
        clip_vision.load = _wrap_with_load_context(
            KIND_CLIP_VISION,
            path_arg_index=0,
            bind_output=lambda result, source_path: result is not None
            and getattr(result, "patcher", None) is not None
            and REGISTRY.bind_object(result.patcher, source_path=source_path, kind=KIND_CLIP_VISION),
        )(original_clip_vision_load)

    original_load_controlnet = _remember_original("controlnet.load_controlnet", controlnet.load_controlnet)
    if controlnet.load_controlnet is original_load_controlnet:
        controlnet.load_controlnet = _wrap_with_load_context(KIND_CONTROLNET, path_arg_index=0)(original_load_controlnet)

    original_load_diffusers = _remember_original("diffusers_load.load_diffusers", diffusers_load.load_diffusers)
    if diffusers_load.load_diffusers is original_load_diffusers:
        diffusers_load.load_diffusers = _wrap_with_load_context(
            KIND_CHECKPOINT,
            path_arg_index=0,
            bind_output=_bind_diffusers_outputs,
        )(original_load_diffusers)

    REGISTRY.refresh_runtime_state()
    _PATCHED = True
    _LOG.info("GPU Resident Loader: monkey patches active on ComfyUI loader and residency paths")
