from __future__ import annotations

import json
import logging
from typing import Any

import comfy.model_management as model_management

from .cleanup import unload_loaded_model
from .external_residency import EXTERNAL_REGISTRY, ensure_external_integrations_installed
from .kj_loader import (
    CheckpointClipLoaderResident,
    CheckpointLoaderResident,
    CheckpointModelLoaderResident,
    CheckpointVAELoaderResident,
    DiffusionModelLoaderResident,
    DiffusionModelSelectorResident,
)
from .residency import REGISTRY

_LOG = logging.getLogger(__name__)


def _entry_report_json(obj: Any) -> str:
    entry = REGISTRY.entry_for_object(obj)
    if entry is None:
        return json.dumps(
            {
                "tracked": False,
                "policy": REGISTRY.get_policy(),
                "message": "Object is not currently bound in the GPU Resident Loader registry.",
            },
            indent=2,
            sort_keys=True,
        )
    REGISTRY.refresh_runtime_state()
    payload = {
        "tracked": True,
        "policy": REGISTRY.get_policy(),
        "entry": entry.as_dict(),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _patcher_for_clip(clip):
    return getattr(clip, "patcher", None)


def _patcher_for_vae(vae):
    return getattr(vae, "patcher", None)


def _preload_patcher(patcher, *, sticky: bool, priority: int) -> None:
    if patcher is None:
        raise RuntimeError("Expected a patcher-capable object, but no patcher was found.")
    model_management.load_models_gpu([patcher], force_full_load=True)
    REGISTRY.set_sticky(patcher, sticky=sticky, priority=priority)
    REGISTRY.touch(patcher)
    REGISTRY.refresh_runtime_state()


def _evict_patcher(patcher, *, unpatch_weights: bool) -> bool:
    if patcher is None:
        raise RuntimeError("Expected a patcher-capable object, but no patcher was found.")
    unloaded = False
    for loaded in list(model_management.current_loaded_models):
        loaded_model = getattr(loaded, "model", None)
        if loaded_model is None:
            continue
        safe_is_clone = False
        if loaded_model is not patcher:
            try:
                safe_is_clone = loaded_model.is_clone(patcher)
            except Exception as exc:
                _LOG.warning("GPU Resident Loader: failed to evaluate clone state during eviction: %s", exc)
        if loaded_model is patcher or safe_is_clone:
            fully_unloaded = unload_loaded_model(
                loaded,
                active_device=getattr(loaded, "device", None),
                force_offload_to_cpu=True,
                unpatch_weights=unpatch_weights,
            )
            if fully_unloaded and unpatch_weights:
                try:
                    model_management.current_loaded_models.remove(loaded)
                except ValueError:
                    pass
            unloaded = unloaded or fully_unloaded
    if unloaded and hasattr(model_management, "soft_empty_cache"):
        model_management.soft_empty_cache()
    REGISTRY.refresh_runtime_state()
    return unloaded


class SetGlobalResidencyPolicy:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "policy": (
                    ["legacy", "balanced", "prefer_gpu", "sticky_gpu"],
                    {"default": "sticky_gpu", "tooltip": "Select the global ingest/offload policy used by the startup patcher."},
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("active_policy",)
    FUNCTION = "set_policy"
    CATEGORY = "GPU Resident Loader/residency"

    def set_policy(self, policy: str):
        return (REGISTRY.set_policy(policy),)


class RegistrySnapshot:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("registry_json",)
    FUNCTION = "snapshot"
    CATEGORY = "GPU Resident Loader/residency"

    def snapshot(self):
        ensure_external_integrations_installed()
        payload = {
            "policy": REGISTRY.get_policy(),
            "entries": REGISTRY.snapshot(),
            "external_entries": EXTERNAL_REGISTRY.snapshot(),
        }
        return (json.dumps(payload, indent=2, sort_keys=True),)


class PinModelResidency:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sticky": ("BOOLEAN", {"default": True}),
                "priority": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "pin"
    CATEGORY = "GPU Resident Loader/residency"

    def pin(self, model, sticky: bool, priority: int):
        REGISTRY.set_sticky(model, sticky=sticky, priority=priority)
        return (model,)


class PinClipResidency:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "sticky": ("BOOLEAN", {"default": True}),
                "priority": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "pin"
    CATEGORY = "GPU Resident Loader/residency"

    def pin(self, clip, sticky: bool, priority: int):
        patcher = _patcher_for_clip(clip)
        if patcher is None:
            raise RuntimeError("Expected a CLIP object with a patcher, but no patcher was found.")
        REGISTRY.set_sticky(patcher, sticky=sticky, priority=priority)
        return (clip,)


class PinVAEResidency:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "sticky": ("BOOLEAN", {"default": True}),
                "priority": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "pin"
    CATEGORY = "GPU Resident Loader/residency"

    def pin(self, vae, sticky: bool, priority: int):
        patcher = _patcher_for_vae(vae)
        if patcher is None:
            raise RuntimeError("Expected a VAE object with a patcher, but no patcher was found.")
        REGISTRY.set_sticky(patcher, sticky=sticky, priority=priority)
        return (vae,)


class PreloadModelToGPU:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sticky": ("BOOLEAN", {"default": True}),
                "priority": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "preload"
    CATEGORY = "GPU Resident Loader/residency"

    def preload(self, model, sticky: bool, priority: int):
        _preload_patcher(model, sticky=sticky, priority=priority)
        return (model,)


class PreloadClipToGPU:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "sticky": ("BOOLEAN", {"default": True}),
                "priority": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "preload"
    CATEGORY = "GPU Resident Loader/residency"

    def preload(self, clip, sticky: bool, priority: int):
        _preload_patcher(_patcher_for_clip(clip), sticky=sticky, priority=priority)
        return (clip,)


class PreloadVAEToGPU:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "sticky": ("BOOLEAN", {"default": True}),
                "priority": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "preload"
    CATEGORY = "GPU Resident Loader/residency"

    def preload(self, vae, sticky: bool, priority: int):
        _preload_patcher(_patcher_for_vae(vae), sticky=sticky, priority=priority)
        return (vae,)


class EvictModelFromGPU:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "unpatch_weights": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "eviction_status")
    FUNCTION = "evict"
    CATEGORY = "GPU Resident Loader/residency"

    def evict(self, model, unpatch_weights: bool):
        unloaded = _evict_patcher(model, unpatch_weights=unpatch_weights)
        return model, ("evicted" if unloaded else "not_loaded")


class EvictClipFromGPU:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "unpatch_weights": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "eviction_status")
    FUNCTION = "evict"
    CATEGORY = "GPU Resident Loader/residency"

    def evict(self, clip, unpatch_weights: bool):
        unloaded = _evict_patcher(_patcher_for_clip(clip), unpatch_weights=unpatch_weights)
        return clip, ("evicted" if unloaded else "not_loaded")


class EvictVAEFromGPU:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "unpatch_weights": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VAE", "STRING")
    RETURN_NAMES = ("vae", "eviction_status")
    FUNCTION = "evict"
    CATEGORY = "GPU Resident Loader/residency"

    def evict(self, vae, unpatch_weights: bool):
        unloaded = _evict_patcher(_patcher_for_vae(vae), unpatch_weights=unpatch_weights)
        return vae, ("evicted" if unloaded else "not_loaded")


class ReportModelResidency:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_json",)
    FUNCTION = "report"
    CATEGORY = "GPU Resident Loader/residency"

    def report(self, model):
        return (_entry_report_json(model),)


class ReportClipResidency:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_json",)
    FUNCTION = "report"
    CATEGORY = "GPU Resident Loader/residency"

    def report(self, clip):
        return (_entry_report_json(_patcher_for_clip(clip)),)


class ReportVAEResidency:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("VAE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_json",)
    FUNCTION = "report"
    CATEGORY = "GPU Resident Loader/residency"

    def report(self, vae):
        return (_entry_report_json(_patcher_for_vae(vae)),)


NODE_CLASS_MAPPINGS = {
    "DiffusionModelSelectorResident": DiffusionModelSelectorResident,
    "DiffusionModelLoaderResident": DiffusionModelLoaderResident,
    "CheckpointLoaderResident": CheckpointLoaderResident,
    "CheckpointModelLoaderResident": CheckpointModelLoaderResident,
    "CheckpointClipLoaderResident": CheckpointClipLoaderResident,
    "CheckpointVAELoaderResident": CheckpointVAELoaderResident,
    "SetGlobalResidencyPolicy": SetGlobalResidencyPolicy,
    "RegistrySnapshot": RegistrySnapshot,
    "PinModelResidency": PinModelResidency,
    "PinClipResidency": PinClipResidency,
    "PinVAEResidency": PinVAEResidency,
    "PreloadModelToGPU": PreloadModelToGPU,
    "PreloadClipToGPU": PreloadClipToGPU,
    "PreloadVAEToGPU": PreloadVAEToGPU,
    "EvictModelFromGPU": EvictModelFromGPU,
    "EvictClipFromGPU": EvictClipFromGPU,
    "EvictVAEFromGPU": EvictVAEFromGPU,
    "ReportModelResidency": ReportModelResidency,
    "ReportClipResidency": ReportClipResidency,
    "ReportVAEResidency": ReportVAEResidency,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionModelSelectorResident": "Diffusion Model Selector Resident",
    "DiffusionModelLoaderResident": "Diffusion Model Loader Resident",
    "CheckpointLoaderResident": "Checkpoint Loader Resident",
    "CheckpointModelLoaderResident": "Checkpoint Model Loader Resident",
    "CheckpointClipLoaderResident": "Checkpoint Clip Loader Resident",
    "CheckpointVAELoaderResident": "Checkpoint VAE Loader Resident",
    "SetGlobalResidencyPolicy": "Set Global Residency Policy",
    "RegistrySnapshot": "Registry Snapshot",
    "PinModelResidency": "Pin Model Residency",
    "PinClipResidency": "Pin CLIP Residency",
    "PinVAEResidency": "Pin VAE Residency",
    "PreloadModelToGPU": "Preload Model To GPU",
    "PreloadClipToGPU": "Preload CLIP To GPU",
    "PreloadVAEToGPU": "Preload VAE To GPU",
    "EvictModelFromGPU": "Evict Model From GPU",
    "EvictClipFromGPU": "Evict CLIP From GPU",
    "EvictVAEFromGPU": "Evict VAE From GPU",
    "ReportModelResidency": "Report Model Residency",
    "ReportClipResidency": "Report CLIP Residency",
    "ReportVAEResidency": "Report VAE Residency",
}
