from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import comfy.model_management as model_management
import torch

from .external_residency import EXTERNAL_REGISTRY, ensure_external_integrations_installed, external_trim_enabled
from .residency import REGISTRY

_ADAPTIVE_HEADROOM_RATIO = 0.125
_ADAPTIVE_HEADROOM_FLOOR_BYTES = 256 * 1024 * 1024
_ADAPTIVE_HEADROOM_CEIL_BYTES = 1024 * 1024 * 1024


def _safe_free_memory(device) -> int:
    return int(model_management.get_free_memory(device))


def _normalize_trim_device(device: str | torch.device | None):
    if device is None or isinstance(device, torch.device):
        return device
    try:
        return torch.device(device)
    except Exception:
        return device


def _safe_is_dead(loaded) -> bool:
    try:
        return loaded.is_dead()
    except Exception:
        return False


def _device_matches(device_a, device_b) -> bool:
    normalized_a = _normalize_trim_device(device_a)
    normalized_b = _normalize_trim_device(device_b)
    if normalized_a is None or normalized_b is None:
        return False
    if isinstance(normalized_a, torch.device) and isinstance(normalized_b, torch.device):
        if normalized_a.type != normalized_b.type:
            return False
        if normalized_a.type == "cuda":
            index_a = 0 if normalized_a.index is None else normalized_a.index
            index_b = 0 if normalized_b.index is None else normalized_b.index
            return index_a == index_b
        return True
    return str(normalized_a) == str(normalized_b)


def _should_force_cpu_offload(model: Any, *, active_device=None, force: bool = False) -> bool:
    if force:
        return True
    active_device = _normalize_trim_device(active_device)
    if not isinstance(active_device, torch.device) or active_device.type != "cuda":
        return False
    return _device_matches(active_device, getattr(model, "offload_device", None))


@contextmanager
def _temporary_offload_device(model: Any, device):
    if model is None or device is None or not hasattr(model, "offload_device"):
        yield
        return

    original_device = model.offload_device
    model.offload_device = device
    try:
        yield
    finally:
        model.offload_device = original_device


def unload_loaded_model(
    loaded,
    *,
    active_device: str | torch.device | None = None,
    force_offload_to_cpu: bool = False,
    unpatch_weights: bool = True,
) -> bool:
    # This helper is intentionally scoped to full unloads owned by this plugin.
    model = getattr(loaded, "model", None)
    if model is None:
        return False

    active_device = _normalize_trim_device(active_device if active_device is not None else getattr(loaded, "device", None))
    force_cpu_offload = _should_force_cpu_offload(
        model,
        active_device=active_device,
        force=force_offload_to_cpu,
    )
    unload_target = torch.device("cpu") if force_cpu_offload else None

    with _temporary_offload_device(model, unload_target):
        return loaded.model_unload(None, unpatch_weights=unpatch_weights)


def _sort_key_for_candidate(entry, *, sticky_respected: bool) -> tuple[int, int, float]:
    if not sticky_respected:
        return (0, 0, getattr(entry, "last_touched", 0.0) if entry is not None else 0.0)
    priority = int(getattr(entry, "priority", 0)) if entry is not None else 0
    last_touched = float(getattr(entry, "last_touched", 0.0)) if entry is not None else 0.0
    return (1, priority, last_touched)


def _should_keep_loaded_model(model: Any, keep_models: tuple[Any, ...]) -> bool:
    if not keep_models:
        return False
    for keep in keep_models:
        if keep is None:
            continue
        if model is keep:
            return True
        is_clone = getattr(model, "is_clone", None)
        if callable(is_clone):
            try:
                if is_clone(keep):
                    return True
            except Exception:
                pass
    return False


def _trim_candidates(
    *,
    device,
    respect_sticky: bool,
    sticky_floor_priority: int,
    keep_models: tuple[Any, ...],
) -> list[tuple[Any, Any, bool, bool]]:
    candidates: list[tuple[Any, Any, bool, bool]] = []
    ensure_external_integrations_installed()
    for loaded in list(model_management.current_loaded_models):
        if device is not None and loaded.device != device:
            continue
        if _safe_is_dead(loaded):
            continue

        model = getattr(loaded, "model", None)
        if model is None:
            continue
        if _should_keep_loaded_model(model, keep_models):
            continue

        entry = REGISTRY.entry_for_object(model)
        sticky_respected = (
            respect_sticky
            and entry is not None
            and bool(getattr(entry, "sticky", False))
            and int(getattr(entry, "priority", 0)) >= int(sticky_floor_priority)
        )
        candidates.append((loaded, entry, sticky_respected, False))

    if external_trim_enabled():
        for external_obj, entry, sticky_respected in EXTERNAL_REGISTRY.candidates(
            device=device,
            respect_sticky=respect_sticky,
            sticky_floor_priority=sticky_floor_priority,
            keep_models=keep_models,
        ):
            candidates.append((external_obj, entry, sticky_respected, True))

    candidates.sort(key=lambda item: _sort_key_for_candidate(item[1], sticky_respected=item[2]))
    return candidates


def trim_resident_vram(
    *,
    device: str | torch.device | None = None,
    target_free_vram_bytes: int,
    respect_sticky: bool,
    sticky_floor_priority: int,
    allow_partial_unload: bool,
    keep_models: tuple[Any, ...] = (),
) -> dict[str, Any]:
    ensure_external_integrations_installed()
    cleanup_models_gc = getattr(model_management, "cleanup_models_gc", None)
    if callable(cleanup_models_gc):
        cleanup_models_gc()

    REGISTRY.refresh_runtime_state()
    EXTERNAL_REGISTRY.refresh_runtime_state()
    device = _normalize_trim_device(device)
    if device is None:
        device = model_management.get_torch_device()
    free_before = _safe_free_memory(device)
    actions: list[dict[str, Any]] = []
    soft_empty_cache = getattr(model_management, "soft_empty_cache", None)
    stopped_reason = "target_met"

    while True:
        free_now = _safe_free_memory(device)
        need = int(target_free_vram_bytes) - free_now
        if need <= 0:
            stopped_reason = "target_met"
            break

        candidates = _trim_candidates(
            device=device,
            respect_sticky=respect_sticky,
            sticky_floor_priority=sticky_floor_priority,
            keep_models=keep_models,
        )
        if not candidates:
            stopped_reason = "no_candidates"
            break

        candidate, entry, sticky_respected, is_external_candidate = candidates[0]
        model = candidate if is_external_candidate else candidate.model
        loaded_before = int(getattr(entry, "loaded_bytes", 0)) if is_external_candidate else int(candidate.model_loaded_memory())
        action = {
            "entry_id": getattr(entry, "entry_id", None),
            "basename": None if entry is None else os.path.basename(getattr(entry, "source_path", "") or ""),
            "tracked": entry is not None,
            "external_candidate": bool(is_external_candidate),
            "sticky_respected": sticky_respected,
            "priority": None if entry is None else int(getattr(entry, "priority", 0)),
            "need_before_bytes": need,
            "loaded_before_bytes": loaded_before,
            "freed_pinned_ram_bytes": 0,
        }

        if (not is_external_candidate and allow_partial_unload and hasattr(model, "pinned_memory_size") and hasattr(model, "partially_unload_ram")):
            try:
                pinned_memory = int(model.pinned_memory_size())
                if pinned_memory > 0:
                    pinned_budget = min(pinned_memory, max(need, 0))
                    model.partially_unload_ram(pinned_budget)
                    action["freed_pinned_ram_bytes"] = pinned_budget
            except Exception as exc:
                action["pinned_ram_warning"] = str(exc)

        if is_external_candidate:
            try:
                fully_unloaded = EXTERNAL_REGISTRY.evict(entry)
                action["mode"] = "external_evict"
            except Exception as exc:
                action["mode"] = "error"
                action["error"] = str(exc)
                actions.append(action)
                stopped_reason = "error"
                break
        else:
            try:
                fully_unloaded = candidate.model_unload(need if allow_partial_unload else None)
                action["mode"] = "full_unload" if fully_unloaded else "partial_unload"
            except Exception as exc:
                if allow_partial_unload:
                    try:
                        fully_unloaded = candidate.model_unload(None)
                        action["mode"] = "full_unload_fallback"
                        action["partial_unload_warning"] = str(exc)
                    except Exception as fallback_exc:
                        action["mode"] = "error"
                        action["error"] = str(fallback_exc)
                        actions.append(action)
                        stopped_reason = "error"
                        break
                else:
                    action["mode"] = "error"
                    action["error"] = str(exc)
                    actions.append(action)
                    stopped_reason = "error"
                    break

        if fully_unloaded and not is_external_candidate:
            try:
                model_management.current_loaded_models.remove(candidate)
            except ValueError:
                pass

        if callable(soft_empty_cache):
            soft_empty_cache()

        REGISTRY.refresh_runtime_state()
        EXTERNAL_REGISTRY.refresh_runtime_state()
        free_after = _safe_free_memory(device)
        if is_external_candidate:
            action["loaded_after_bytes"] = 0 if fully_unloaded else int(getattr(entry, "loaded_bytes", 0))
        else:
            action["loaded_after_bytes"] = 0 if fully_unloaded else int(candidate.model_loaded_memory())
        action["freed_vram_bytes"] = max(0, free_after - free_now)
        action["free_after_bytes"] = free_after
        actions.append(action)

        if action["freed_vram_bytes"] <= 0 and not fully_unloaded:
            stopped_reason = "no_progress"
            break

    free_after = _safe_free_memory(device)
    target_met = free_after >= int(target_free_vram_bytes)
    if target_met:
        stopped_reason = "target_met"
    return {
        "status": "met_target" if target_met else ("error" if stopped_reason == "error" else "partial"),
        "stopped_reason": stopped_reason,
        "target_met": target_met,
        "device": str(device),
        "target_free_vram_bytes": int(target_free_vram_bytes),
        "free_before_bytes": free_before,
        "free_after_bytes": free_after,
        "freed_vram_bytes": max(0, free_after - free_before),
        "respect_sticky": bool(respect_sticky),
        "sticky_floor_priority": int(sticky_floor_priority),
        "allow_partial_unload": bool(allow_partial_unload),
        "external_trim_enabled": bool(external_trim_enabled()),
        "actions": actions,
    }


def adaptive_headroom_bytes(required_bytes: int) -> int:
    required = max(0, int(required_bytes))
    if required == 0:
        return 0
    return min(
        _ADAPTIVE_HEADROOM_CEIL_BYTES,
        max(_ADAPTIVE_HEADROOM_FLOOR_BYTES, int(required * _ADAPTIVE_HEADROOM_RATIO)),
    )


def trim_resident_vram_for_load(
    *,
    required_bytes: int,
    reason: str,
    device: str | torch.device | None = None,
    respect_sticky: bool = True,
    sticky_floor_priority: int = 0,
    allow_partial_unload: bool = True,
    keep_models: tuple[Any, ...] = (),
) -> dict[str, Any]:
    estimated_load_bytes = max(0, int(required_bytes))
    headroom_bytes = adaptive_headroom_bytes(estimated_load_bytes)
    target_free_vram_bytes = estimated_load_bytes + headroom_bytes

    report = trim_resident_vram(
        device=device,
        target_free_vram_bytes=target_free_vram_bytes,
        respect_sticky=respect_sticky,
        sticky_floor_priority=sticky_floor_priority,
        allow_partial_unload=allow_partial_unload,
        keep_models=keep_models,
    )
    report["trim_strategy"] = "adaptive_load_request"
    report["trim_reason"] = str(reason)
    report["estimated_load_bytes"] = estimated_load_bytes
    report["adaptive_headroom_bytes"] = headroom_bytes
    report["kept_loaded_models"] = len([model for model in keep_models if model is not None])
    return report
