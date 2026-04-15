from __future__ import annotations

import functools
import json
import logging
import os
import struct
from typing import Any, Callable

import torch
from safetensors import safe_open

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


def _wrap_load_models_gpu(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(models, *args, **kwargs):
        result = func(models, *args, **kwargs)
        for model in list(models):
            REGISTRY.touch(model)
        REGISTRY.refresh_runtime_state()
        return result

    return wrapper


def _wrap_free_memory(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(memory_required, device, keep_loaded=None, *args, **kwargs):
        import comfy.model_management as model_management

        keep_loaded = list(keep_loaded or [])
        sticky_wrappers = []
        if REGISTRY.get_policy() == "sticky_gpu":
            sticky_wrappers = [w for w in REGISTRY.sticky_loaded_wrappers(device) if w not in keep_loaded]

        protected_wrappers: list[Any] = []
        if device is not None and sticky_wrappers:
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
                    free_now + sum(max(0, loaded.model_loaded_memory()) for loaded in unloadable_wrappers) - memory_required,
                )
                protected_memory = 0
                for loaded in sticky_wrappers:
                    estimated_memory = max(0, loaded.model_loaded_memory())
                    if protected_memory + estimated_memory <= available_for_protection:
                        protected_wrappers.append(loaded)
                        protected_memory += estimated_memory

        unloaded = func(memory_required, device, keep_loaded + protected_wrappers, *args, **kwargs)

        REGISTRY.refresh_runtime_state()
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


def install_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import comfy.clip_vision as clip_vision
    import comfy.controlnet as controlnet
    import comfy.diffusers_load as diffusers_load
    import comfy.model_management as model_management
    import comfy.sd as comfy_sd
    import comfy.utils as comfy_utils

    original_load_torch_file = _remember_original("utils.load_torch_file", comfy_utils.load_torch_file)
    if comfy_utils.load_torch_file is original_load_torch_file:
        comfy_utils.load_torch_file = _patched_load_torch_file
    if hasattr(clip_vision, "load_torch_file"):
        original_clip_vision_load_torch_file = _remember_original("clip_vision.load_torch_file", clip_vision.load_torch_file)
        if clip_vision.load_torch_file is original_clip_vision_load_torch_file:
            clip_vision.load_torch_file = comfy_utils.load_torch_file

    _patch_model_management_devices()

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
