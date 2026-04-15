from __future__ import annotations

from collections.abc import Mapping
import contextlib
import dataclasses
import inspect
import logging
import os
import sys
import threading
import time
import weakref
from typing import Any, Callable

import torch

from .residency import KIND_MODEL, KIND_VAE, REGISTRY

_LOG = logging.getLogger(__name__)
_SEEDVR2_PATCHED_CLASS_IDS: set[int] = set()
_SEEDVR2_PATCHING_CLASS_IDS: set[int] = set()
_SEEDVR2_PATCHING_THREAD_IDS: dict[int, int] = {}
_SEEDVR2_PATCH_LOCK = threading.Lock()
_SEEDVR2_PATCH_CONDITION = threading.Condition(_SEEDVR2_PATCH_LOCK)


def _now() -> float:
    return time.time()


def _normalize_device(device: str | torch.device | None):
    if device is None or isinstance(device, torch.device):
        return device
    try:
        return torch.device(device)
    except Exception:
        return device


def _device_matches(device_a, device_b) -> bool:
    normalized_a = _normalize_device(device_a)
    normalized_b = _normalize_device(device_b)
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


def _iter_seedvr2_wrapper_chain(model: Any):
    if model is None:
        return

    stack = [model]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        yield current
        for attr in ("_orig_mod", "dit_model"):
            child = getattr(current, attr, None)
            if child is not None:
                stack.append(child)


def _seedvr2_is_claimed(model: Any) -> bool:
    return any(bool(getattr(current, "_seedvr2_cache_claimed", False)) for current in _iter_seedvr2_wrapper_chain(model))


def _first_tensor_device(model: Any) -> str | None:
    if model is None:
        return None
    try:
        for param in model.parameters():
            return str(param.device)
    except Exception:
        pass
    try:
        for buffer in model.buffers():
            return str(buffer.device)
    except Exception:
        pass
    return None


def _unique_tensor_nbytes(model: Any) -> int:
    if model is None:
        return 0

    total = 0
    seen_storages: set[tuple[int, int, int]] = set()

    def visit_tensor(tensor: torch.Tensor) -> None:
        nonlocal total
        if tensor is None:
            return
        try:
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes(), int(tensor.device.index or 0) if tensor.device.type == "cuda" else -1)
        except Exception:
            key = (id(tensor), tensor.numel() * tensor.element_size(), -2)
        if key in seen_storages:
            return
        seen_storages.add(key)
        total += int(tensor.numel()) * int(tensor.element_size())

    try:
        for tensor in model.parameters():
            visit_tensor(tensor)
    except Exception:
        pass
    try:
        for tensor in model.buffers():
            visit_tensor(tensor)
    except Exception:
        pass
    return int(total)


def _seedvr2_entry_key(kind: str, node_id: Any) -> str:
    return f"seedvr2:{kind}:{node_id}"


def _seedvr2_source_path(kind: str, node_id: Any, config: dict[str, Any], model: Any) -> str:
    model_name = config.get("model") or getattr(model, "_model_name", None) or f"node_{node_id}"
    return f"seedvr2/{kind}/{node_id}/{model_name}"


def _seedvr2_state_provider(
    model_ref: Callable[[], Any | None],
    config: dict[str, Any],
) -> Callable[[], dict[str, Any]]:
    def provider() -> dict[str, Any]:
        model = model_ref()
        if model is None:
            return {}
        total_bytes = _unique_tensor_nbytes(model)
        return {
            "current_device": _first_tensor_device(model),
            "load_device": config.get("device"),
            "offload_device": config.get("offload_device"),
            "claimed": _seedvr2_is_claimed(model),
            "loaded_bytes": total_bytes,
            "total_bytes": total_bytes,
        }

    return provider


def _coerce_external_bytes(value: Any, fallback: int, *, cache_key: str, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        _LOG.debug(
            "GPU Resident Loader: ignoring invalid external %s for %s: %r (%s)",
            field_name,
            cache_key,
            value,
            exc,
        )
        return int(fallback)


@dataclasses.dataclass(slots=True)
class ExternalResidencyEntry:
    entry_id: str
    cache_key: str
    kind: str
    source_path: str
    sticky: bool
    priority: int
    created_at: float = dataclasses.field(default_factory=_now)
    last_touched: float = dataclasses.field(default_factory=_now)
    loaded_bytes: int = 0
    total_bytes: int = 0
    load_device: str | None = None
    offload_device: str | None = None
    current_device: str | None = None
    claimed: bool = False
    notes: list[str] = dataclasses.field(default_factory=list)
    object_ref: weakref.ReferenceType[Any] | None = None
    state_provider: Callable[[], dict[str, Any]] | None = None
    evict_callback: Callable[[], bool] | None = None

    def object(self) -> Any | None:
        return None if self.object_ref is None else self.object_ref()

    def is_alive(self) -> bool:
        return self.object() is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "cache_key": self.cache_key,
            "kind": self.kind,
            "source_path": self.source_path,
            "basename": os.path.basename(self.source_path) if self.source_path else None,
            "sticky": self.sticky,
            "priority": self.priority,
            "created_at": self.created_at,
            "last_touched": self.last_touched,
            "loaded_bytes": self.loaded_bytes,
            "total_bytes": self.total_bytes,
            "load_device": self.load_device,
            "offload_device": self.offload_device,
            "current_device": self.current_device,
            "claimed": self.claimed,
            "notes": list(self.notes),
            "alive": self.is_alive(),
            "external": True,
        }


class ExternalResidencyRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, ExternalResidencyEntry] = {}
        self._cache_key_to_entry: dict[str, str] = {}
        self._next_entry_seq = 1

    def _make_entry_id(self, kind: str, source_path: str) -> str:
        basename = os.path.basename(source_path) or "anonymous"
        entry_id = f"external:{kind}:{basename}:{self._next_entry_seq}"
        self._next_entry_seq += 1
        return entry_id

    def bind(
        self,
        *,
        cache_key: str,
        obj: Any,
        kind: str,
        source_path: str,
        state_provider: Callable[[], dict[str, Any]],
        evict_callback: Callable[[], bool],
        sticky: bool = False,
        priority: int | None = None,
        note: str | None = None,
    ) -> ExternalResidencyEntry:
        if obj is None:
            raise ValueError("Cannot bind None into external residency registry")

        with self._lock:
            entry_id = self._cache_key_to_entry.get(cache_key)
            if entry_id is not None and entry_id in self._entries:
                entry = self._entries[entry_id]
            else:
                entry_id = self._make_entry_id(kind, source_path)
                entry = ExternalResidencyEntry(
                    entry_id=entry_id,
                    cache_key=cache_key,
                    kind=kind,
                    source_path=source_path,
                    sticky=bool(sticky),
                    priority=REGISTRY.default_priority(kind) if priority is None else int(priority),
                )
                self._entries[entry_id] = entry
                self._cache_key_to_entry[cache_key] = entry_id

            try:
                entry.object_ref = weakref.ref(obj)
            except TypeError:
                entry.object_ref = None
            entry.cache_key = cache_key
            entry.kind = kind
            entry.source_path = source_path
            entry.sticky = bool(sticky)
            entry.priority = REGISTRY.default_priority(kind) if priority is None else int(priority)
            entry.state_provider = state_provider
            entry.evict_callback = evict_callback
            entry.last_touched = _now()
            if note and note not in entry.notes:
                entry.notes.append(note)

        self.refresh_runtime_state()
        return entry

    def remove(self, *, cache_key: str) -> bool:
        with self._lock:
            entry_id = self._cache_key_to_entry.pop(cache_key, None)
            if entry_id is None:
                return False
            return self._entries.pop(entry_id, None) is not None

    def refresh_runtime_state(self) -> None:
        ensure_external_integrations_installed()
        stale_keys: list[str] = []
        with self._lock:
            for cache_key, entry_id in list(self._cache_key_to_entry.items()):
                entry = self._entries.get(entry_id)
                if entry is None:
                    stale_keys.append(cache_key)
                    continue
                obj = entry.object()
                if obj is None:
                    stale_keys.append(cache_key)
                    continue
                state_provider = entry.state_provider
                if state_provider is None:
                    continue
                try:
                    state = state_provider() or {}
                except Exception as exc:
                    _LOG.debug("GPU Resident Loader: failed to refresh external cache state for %s: %s", cache_key, exc)
                    continue
                entry.current_device = state.get("current_device")
                entry.load_device = state.get("load_device")
                entry.offload_device = state.get("offload_device")
                entry.claimed = bool(state.get("claimed", False))
                entry.loaded_bytes = _coerce_external_bytes(
                    state.get("loaded_bytes", entry.loaded_bytes or 0),
                    entry.loaded_bytes or 0,
                    cache_key=cache_key,
                    field_name="loaded_bytes",
                )
                entry.total_bytes = _coerce_external_bytes(
                    state.get("total_bytes", entry.total_bytes or entry.loaded_bytes or 0),
                    entry.total_bytes or entry.loaded_bytes or 0,
                    cache_key=cache_key,
                    field_name="total_bytes",
                )
            for cache_key in stale_keys:
                entry_id = self._cache_key_to_entry.pop(cache_key, None)
                if entry_id is not None:
                    self._entries.pop(entry_id, None)

    def candidates(
        self,
        *,
        device: str | torch.device | None,
        respect_sticky: bool,
        sticky_floor_priority: int,
        keep_models: tuple[Any, ...],
    ) -> list[tuple[Any, ExternalResidencyEntry, bool]]:
        self.refresh_runtime_state()
        output: list[tuple[Any, ExternalResidencyEntry, bool]] = []
        with self._lock:
            for entry in self._entries.values():
                obj = entry.object()
                if obj is None:
                    continue
                if entry.claimed:
                    continue
                if device is not None and not _device_matches(device, entry.current_device):
                    continue
                if any(obj is keep for keep in keep_models if keep is not None):
                    continue
                sticky_respected = (
                    respect_sticky
                    and bool(entry.sticky)
                    and int(entry.priority) >= int(sticky_floor_priority)
                )
                output.append((obj, entry, sticky_respected))
        return output

    def evict(self, entry: ExternalResidencyEntry) -> bool:
        callback = entry.evict_callback
        if callback is None:
            return False
        try:
            result = bool(callback())
        except Exception as exc:
            _LOG.warning(
                "GPU Resident Loader: external eviction callback failed for %s: %s",
                entry.cache_key,
                exc,
            )
            result = False
        finally:
            self.refresh_runtime_state()
        return result

    def snapshot(self) -> list[dict[str, Any]]:
        self.refresh_runtime_state()
        with self._lock:
            items = [entry.as_dict() for entry in self._entries.values()]
        items.sort(
            key=lambda item: (
                not item["sticky"],
                item["kind"],
                item["basename"] or "",
            )
        )
        return items


EXTERNAL_REGISTRY = ExternalResidencyRegistry()


def external_trim_enabled() -> bool:
    value = os.environ.get("COMFYUI_GPU_RESIDENT_TRIM_EXTERNAL", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _call_seedvr2_method_with_optional_expected_model(
    method: Callable[..., Any],
    *args: Any,
    debug: Any = None,
    expected_model: Any = None,
) -> Any:
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    kwargs: dict[str, Any] = {}
    if accepts_kwargs or "debug" in parameters:
        kwargs["debug"] = debug
    if expected_model is not None and (accepts_kwargs or "expected_model" in parameters):
        kwargs["expected_model"] = expected_model
    return method(*args, **kwargs)


def _register_seedvr2_cached_model(global_cache: Any, *, kind: str, config: Any, model: Any) -> None:
    if not isinstance(config, Mapping):
        _LOG.debug(
            "GPU Resident Loader: skipping SeedVR2 %s cache entry with unexpected config type: %s",
            kind,
            type(config).__name__,
        )
        return

    node_id = config.get("node_id")
    if node_id is None or model is None:
        return

    try:
        model_ref = weakref.ref(model)
    except TypeError:
        def model_ref() -> Any | None:
            return None

    source_path = _seedvr2_source_path(kind, node_id, config, model)
    cache_key = _seedvr2_entry_key(kind, node_id)
    registry_kind = KIND_MODEL if kind == "dit" else KIND_VAE

    if kind == "dit":
        def evict_callback() -> bool:
            return bool(
                _call_seedvr2_method_with_optional_expected_model(
                    global_cache.remove_dit,
                    {"node_id": node_id},
                    debug=None,
                    expected_model=model_ref(),
                )
            )
        note = f"SeedVR2 cached DiT node {node_id}"
    else:
        def evict_callback() -> bool:
            return bool(
                _call_seedvr2_method_with_optional_expected_model(
                    global_cache.remove_vae,
                    {"node_id": node_id},
                    debug=None,
                    expected_model=model_ref(),
                )
            )
        note = f"SeedVR2 cached VAE node {node_id}"

    EXTERNAL_REGISTRY.bind(
        cache_key=cache_key,
        obj=model,
        kind=registry_kind,
        source_path=source_path,
        state_provider=_seedvr2_state_provider(model_ref, config),
        evict_callback=evict_callback,
        sticky=False,
        priority=REGISTRY.default_priority(registry_kind),
        note=note,
    )


def _install_seedvr2_integration_for_module(module: Any) -> bool:
    model_cache_cls = getattr(module, "GlobalModelCache", None)
    get_global_cache = getattr(module, "get_global_cache", None)
    if model_cache_cls is None or not callable(get_global_cache):
        return False

    class_id = id(model_cache_cls)
    thread_id = threading.get_ident()
    original_set_dit = None
    original_set_vae = None
    original_replace_dit = None
    original_replace_vae = None
    original_remove_dit = None
    original_remove_vae = None
    provisional_cache_keys: set[str] = set()

    try:
        with _SEEDVR2_PATCH_CONDITION:
            while (
                class_id in _SEEDVR2_PATCHING_CLASS_IDS
                and _SEEDVR2_PATCHING_THREAD_IDS.get(class_id) != thread_id
            ):
                _SEEDVR2_PATCH_CONDITION.wait()
            if class_id in _SEEDVR2_PATCHED_CLASS_IDS or _SEEDVR2_PATCHING_THREAD_IDS.get(class_id) == thread_id:
                return True
            _SEEDVR2_PATCHING_CLASS_IDS.add(class_id)
            _SEEDVR2_PATCHING_THREAD_IDS[class_id] = thread_id

            original_set_dit = model_cache_cls.set_dit
            original_set_vae = model_cache_cls.set_vae
            original_replace_dit = getattr(model_cache_cls, "replace_dit", None)
            original_replace_vae = getattr(model_cache_cls, "replace_vae", None)
            original_remove_dit = model_cache_cls.remove_dit
            original_remove_vae = model_cache_cls.remove_vae

        def set_dit_wrapper(self, dit_config, model, model_name, debug=None):
            result = original_set_dit(self, dit_config, model, model_name, debug)
            if result is not None:
                _register_seedvr2_cached_model(self, kind="dit", config=dit_config, model=model)
            return result

        def set_vae_wrapper(self, vae_config, model, model_name, debug=None):
            result = original_set_vae(self, vae_config, model, model_name, debug)
            if result is not None:
                _register_seedvr2_cached_model(self, kind="vae", config=vae_config, model=model)
            return result

        def replace_dit_wrapper(self, dit_config, model, debug=None, expected_model=None):
            if original_replace_dit is None:
                return False
            result = _call_seedvr2_method_with_optional_expected_model(
                original_replace_dit,
                self,
                dit_config,
                model,
                debug=debug,
                expected_model=expected_model,
            )
            if result:
                _register_seedvr2_cached_model(self, kind="dit", config=dit_config, model=model)
            return result

        def replace_vae_wrapper(self, vae_config, model, debug=None, expected_model=None):
            if original_replace_vae is None:
                return False
            result = _call_seedvr2_method_with_optional_expected_model(
                original_replace_vae,
                self,
                vae_config,
                model,
                debug=debug,
                expected_model=expected_model,
            )
            if result:
                _register_seedvr2_cached_model(self, kind="vae", config=vae_config, model=model)
            return result

        def remove_dit_wrapper(self, dit_config, debug=None, expected_model=None):
            result = _call_seedvr2_method_with_optional_expected_model(
                original_remove_dit,
                self,
                dit_config,
                debug=debug,
                expected_model=expected_model,
            )
            if result:
                EXTERNAL_REGISTRY.remove(cache_key=_seedvr2_entry_key("dit", dit_config.get("node_id")))
            return result

        def remove_vae_wrapper(self, vae_config, debug=None, expected_model=None):
            result = _call_seedvr2_method_with_optional_expected_model(
                original_remove_vae,
                self,
                vae_config,
                debug=debug,
                expected_model=expected_model,
            )
            if result:
                EXTERNAL_REGISTRY.remove(cache_key=_seedvr2_entry_key("vae", vae_config.get("node_id")))
            return result

        global_cache = get_global_cache()
        model_cache_lock = getattr(global_cache, "_model_cache_lock", None)
        lock_context = model_cache_lock if model_cache_lock is not None else contextlib.nullcontext()
        with lock_context:
            model_cache_cls.set_dit = set_dit_wrapper
            model_cache_cls.set_vae = set_vae_wrapper
            if original_replace_dit is not None:
                model_cache_cls.replace_dit = replace_dit_wrapper
            if original_replace_vae is not None:
                model_cache_cls.replace_vae = replace_vae_wrapper
            model_cache_cls.remove_dit = remove_dit_wrapper
            model_cache_cls.remove_vae = remove_vae_wrapper
            dit_items = list(getattr(global_cache, "_dit_models", {}).items())
            vae_items = list(getattr(global_cache, "_vae_models", {}).items())
            for _node_id, entry in dit_items:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    continue
                model, config = entry
                if model is not None:
                    if isinstance(config, Mapping) and config.get("node_id") is not None:
                        provisional_cache_keys.add(_seedvr2_entry_key("dit", config.get("node_id")))
                    _register_seedvr2_cached_model(global_cache, kind="dit", config=config, model=model)
            for _node_id, entry in vae_items:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    continue
                model, config = entry
                if model is not None:
                    if isinstance(config, Mapping) and config.get("node_id") is not None:
                        provisional_cache_keys.add(_seedvr2_entry_key("vae", config.get("node_id")))
                    _register_seedvr2_cached_model(global_cache, kind="vae", config=config, model=model)
        with _SEEDVR2_PATCH_CONDITION:
            _SEEDVR2_PATCHING_CLASS_IDS.discard(class_id)
            _SEEDVR2_PATCHED_CLASS_IDS.add(class_id)
            _SEEDVR2_PATCHING_THREAD_IDS.pop(class_id, None)
            _SEEDVR2_PATCH_CONDITION.notify_all()
    except Exception:
        _LOG.debug(
            "GPU Resident Loader: rolling back SeedVR2 integration for class_id=%s provisional_cache_keys=%s",
            class_id,
            sorted(provisional_cache_keys),
            exc_info=True,
        )
        for cache_key in provisional_cache_keys:
            EXTERNAL_REGISTRY.remove(cache_key=cache_key)
        with _SEEDVR2_PATCH_CONDITION:
            if original_set_dit is not None:
                model_cache_cls.set_dit = original_set_dit
            if original_set_vae is not None:
                model_cache_cls.set_vae = original_set_vae
            if original_replace_dit is not None:
                model_cache_cls.replace_dit = original_replace_dit
            if original_replace_vae is not None:
                model_cache_cls.replace_vae = original_replace_vae
            if original_remove_dit is not None:
                model_cache_cls.remove_dit = original_remove_dit
            if original_remove_vae is not None:
                model_cache_cls.remove_vae = original_remove_vae
            _SEEDVR2_PATCHING_CLASS_IDS.discard(class_id)
            _SEEDVR2_PATCHED_CLASS_IDS.discard(class_id)
            _SEEDVR2_PATCHING_THREAD_IDS.pop(class_id, None)
            _SEEDVR2_PATCH_CONDITION.notify_all()
        raise

    _LOG.info("GPU Resident Loader: integrated external SeedVR2 cache visibility hooks")
    return True


def ensure_external_integrations_installed() -> None:
    for module in list(sys.modules.values()):
        if module is None:
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        normalized_file = os.path.abspath(module_file).replace("\\", "/")
        if not normalized_file.endswith("/src/core/model_cache.py"):
            continue
        if "seedvr2" not in normalized_file.lower():
            continue
        if hasattr(module, "GlobalModelCache") and hasattr(module, "get_global_cache"):
            try:
                _install_seedvr2_integration_for_module(module)
            except Exception as exc:
                _LOG.warning(
                    "GPU Resident Loader: failed to install SeedVR2 external integration from %s: %s",
                    normalized_file,
                    exc,
                )
