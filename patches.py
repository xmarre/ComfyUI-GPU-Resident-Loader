from __future__ import annotations

import functools
import logging
import os
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
) -> torch.Tensor:
    if _tensor_key_requires_cpu(key):
        return _copy_tensor_if_needed(tensor, torch.device("cpu"))

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
            if comfy.memory_management.aimdo_enabled and requested_device.type == "cpu":
                sd, metadata = comfy_utils.load_safetensors(ckpt)
                method = "safetensors_aimdo_cpu"
                if not return_metadata:
                    metadata = None
                _record_generic_load(
                    path=ckpt,
                    method=method,
                    requested_device=requested_device,
                    actual_device="cpu",
                )
                return (sd, metadata) if return_metadata else sd

            safe_device = _safe_open_device_arg(requested_device)
            with safe_open(ckpt, framework="pt", device=safe_device) as handle:
                sd = {}
                disable_mmap = getattr(comfy_utils, "DISABLE_MMAP", False)
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    sd[key] = _prepare_loaded_tensor(
                        key,
                        tensor,
                        requested_device,
                        disable_mmap=disable_mmap,
                    )
                if return_metadata:
                    metadata = handle.metadata()

            actual_device = _state_dict_device_summary(sd, requested_device)
            method = "safetensors_gpu_direct" if requested_device.type == "cuda" else "safetensors_cpu"
            _record_generic_load(
                path=ckpt,
                method=method,
                requested_device=requested_device,
                actual_device=actual_device,
            )
            return (sd, metadata) if return_metadata else sd
        except Exception as exc:
            if requested_device.type == "cuda":
                _LOG.warning(
                    "GPU Resident Loader: direct GPU safetensors load failed for %s; falling back to CPU path: %s",
                    ckpt,
                    exc,
                )
                try:
                    with safe_open(ckpt, framework="pt", device="cpu") as handle:
                        sd = {}
                        for key in handle.keys():
                            sd[key] = _prepare_loaded_tensor(
                                key,
                                handle.get_tensor(key).to(requested_device),
                                requested_device,
                                disable_mmap=False,
                            )
                        if return_metadata:
                            metadata = handle.metadata()
                    _record_generic_load(
                        path=ckpt,
                        method="safetensors_cpu_then_copy_to_cuda",
                        requested_device=requested_device,
                        actual_device=_state_dict_device_summary(sd, requested_device),
                        error=str(exc),
                    )
                    return (sd, metadata) if return_metadata else sd
                except Exception as fallback_exc:
                    _record_generic_load(
                        path=ckpt,
                        method="safetensors_cpu_fallback_failed",
                        requested_device=requested_device,
                        actual_device="error",
                        error=str(fallback_exc),
                    )
                    raise fallback_exc from exc

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

        unloaded = func(memory_required, device, keep_loaded + sticky_wrappers, *args, **kwargs)

        if device is not None and sticky_wrappers:
            try:
                free_after = model_management.get_free_memory(device)
            except Exception:
                free_after = None
            if free_after is not None and free_after < memory_required:
                _LOG.warning(
                    "GPU Resident Loader: sticky set exceeded VRAM budget; allowing fallback eviction to satisfy request"
                )
                unloaded = func(memory_required, device, keep_loaded, *args, **kwargs)

        REGISTRY.refresh_runtime_state()
        return unloaded

    return wrapper


def _remember_original(key: str, value: Callable[..., Any]) -> Callable[..., Any]:
    return _ORIGINALS.setdefault(key, value)


def _patch_model_management_devices() -> None:
    import comfy.model_management as model_management

    def wrap_device_func(name: str) -> None:
        key = f"model_management.{name}"
        original = _remember_original(key, getattr(model_management, name))
        if getattr(model_management, name) is not original:
            return

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            if not REGISTRY.wants_gpu_offload(name):
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
