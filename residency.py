from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import json
import logging
import os
import threading
import time
import weakref
from collections.abc import Iterator
from typing import Any

import torch


_LOG = logging.getLogger(__name__)
_LOAD_CONTEXT: contextvars.ContextVar[LoadContext | None] = contextvars.ContextVar(
    "gpu_resident_loader_load_context",
    default=None,
)

POLICIES = ("legacy", "balanced", "prefer_gpu", "sticky_gpu")
KIND_MODEL = "model"
KIND_CLIP = "clip"
KIND_VAE = "vae"
KIND_CHECKPOINT = "checkpoint"
KIND_CLIP_VISION = "clip_vision"
KIND_CONTROLNET = "controlnet"


def _now() -> float:
    return time.time()


@dataclasses.dataclass(slots=True)
class LoadContext:
    kind: str
    source_path: str | None = None
    explicit_device: torch.device | None = None
    note: str | None = None


@dataclasses.dataclass(slots=True)
class LoadReport:
    path: str
    kind: str
    method: str
    requested_device: str
    actual_device: str
    timestamp: float = dataclasses.field(default_factory=_now)
    note: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "method": self.method,
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "timestamp": self.timestamp,
            "note": self.note,
            "error": self.error,
        }


@dataclasses.dataclass(slots=True)
class ResidencyEntry:
    entry_id: str
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
    last_method: str | None = None
    last_report: dict[str, Any] | None = None
    notes: list[str] = dataclasses.field(default_factory=list)
    object_ref: weakref.ReferenceType[Any] | None = None

    def is_alive(self) -> bool:
        return self.object_ref is not None and self.object_ref() is not None

    def object(self) -> Any | None:
        return None if self.object_ref is None else self.object_ref()

    def as_dict(self) -> dict[str, Any]:
        basename = os.path.basename(self.source_path) if self.source_path else None
        return {
            "entry_id": self.entry_id,
            "kind": self.kind,
            "source_path": self.source_path,
            "basename": basename,
            "sticky": self.sticky,
            "priority": self.priority,
            "created_at": self.created_at,
            "last_touched": self.last_touched,
            "loaded_bytes": self.loaded_bytes,
            "total_bytes": self.total_bytes,
            "load_device": self.load_device,
            "offload_device": self.offload_device,
            "current_device": self.current_device,
            "last_method": self.last_method,
            "last_report": self.last_report,
            "notes": list(self.notes),
            "alive": self.is_alive(),
        }


class ResidencyRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, ResidencyEntry] = {}
        self._reports_by_path: dict[str, LoadReport] = {}
        self._path_to_entry: dict[tuple[str, str], str] = {}
        self._policy = self._default_policy()

    def _default_policy(self) -> str:
        env_value = os.environ.get("COMFYUI_GPU_RESIDENT_POLICY", "").strip().lower()
        if env_value in POLICIES:
            return env_value

        try:
            from comfy.cli_args import args
        except Exception:
            return "prefer_gpu"

        if getattr(args, "gpu_only", False):
            return "sticky_gpu"
        if getattr(args, "highvram", False):
            return "sticky_gpu"
        return "prefer_gpu"

    def get_policy(self) -> str:
        with self._lock:
            return self._policy

    def set_policy(self, policy: str) -> str:
        normalized = str(policy).strip().lower()
        if normalized not in POLICIES:
            raise ValueError(f"Unsupported residency policy: {policy}")
        with self._lock:
            self._policy = normalized
        _LOG.info("GPU Resident Loader: policy set to %s", normalized)
        return normalized

    def wants_gpu_ingest(self, kind: str | None = None) -> bool:
        policy = self.get_policy()
        return policy in {"prefer_gpu", "sticky_gpu"}

    def wants_gpu_offload(self, kind: str | None = None) -> bool:
        policy = self.get_policy()
        return policy in {"prefer_gpu", "sticky_gpu"}

    def autopin_on_bind(self, kind: str | None = None) -> bool:
        return self.get_policy() == "sticky_gpu"

    def explicit_load_device(self, kind: str, source_path: str | None = None) -> torch.device | None:
        if not self.wants_gpu_ingest(kind):
            return None
        try:
            import comfy.model_management as model_management
        except Exception:
            return None
        dev = model_management.get_torch_device()
        if getattr(dev, "type", None) == "cpu":
            return None
        return dev

    @contextlib.contextmanager
    def load_context(
        self,
        *,
        kind: str,
        source_path: str | None = None,
        explicit_device: torch.device | None = None,
        note: str | None = None,
    ) -> Iterator[None]:
        token = _LOAD_CONTEXT.set(
            LoadContext(
                kind=kind,
                source_path=source_path,
                explicit_device=explicit_device,
                note=note,
            )
        )
        try:
            yield
        finally:
            _LOAD_CONTEXT.reset(token)

    def current_context(self) -> LoadContext | None:
        return _LOAD_CONTEXT.get()

    def record_load(
        self,
        *,
        path: str,
        kind: str,
        method: str,
        requested_device: str,
        actual_device: str,
        note: str | None = None,
        error: str | None = None,
    ) -> LoadReport:
        report = LoadReport(
            path=path,
            kind=kind,
            method=method,
            requested_device=requested_device,
            actual_device=actual_device,
            note=note,
            error=error,
        )
        with self._lock:
            self._reports_by_path[path] = report
            entry_id = self._path_to_entry.get((kind, path))
            if entry_id is not None:
                entry = self._entries.get(entry_id)
                if entry is not None:
                    entry.last_method = method
                    entry.last_report = report.as_dict()
                    entry.last_touched = _now()
                    entry.current_device = actual_device
        return report

    def latest_report_for_path(self, path: str | None) -> LoadReport | None:
        if not path:
            return None
        with self._lock:
            return self._reports_by_path.get(path)

    def _make_entry_id(self, kind: str, source_path: str) -> str:
        basename = os.path.basename(source_path) or "anonymous"
        return f"{kind}:{basename}:{len(self._entries) + 1}"

    def bind_object(
        self,
        obj: Any,
        *,
        source_path: str,
        kind: str,
        sticky: bool | None = None,
        priority: int = 0,
        note: str | None = None,
    ) -> ResidencyEntry:
        if obj is None:
            raise ValueError("Cannot bind None into residency registry")

        with self._lock:
            entry_id = getattr(obj, "__gpu_resident_loader_entry_id__", None)
            if entry_id is not None and entry_id in self._entries:
                entry = self._entries[entry_id]
            else:
                entry_id = self._make_entry_id(kind, source_path)
                entry = ResidencyEntry(
                    entry_id=entry_id,
                    kind=kind,
                    source_path=source_path,
                    sticky=self.autopin_on_bind(kind) if sticky is None else bool(sticky),
                    priority=int(priority),
                )
                self._entries[entry_id] = entry
                self._path_to_entry[(kind, source_path)] = entry_id
                try:
                    setattr(obj, "__gpu_resident_loader_entry_id__", entry_id)
                except (AttributeError, TypeError):
                    _LOG.debug(
                        "GPU Resident Loader: could not tag object %r with residency entry id %s",
                        type(obj),
                        entry_id,
                    )

            try:
                entry.object_ref = weakref.ref(obj)
            except TypeError:
                entry.object_ref = None
                _LOG.debug(
                    "GPU Resident Loader: object %r is not weak-referenceable; tracking metadata only",
                    type(obj),
                )
            entry.sticky = entry.sticky if sticky is None else bool(sticky)
            entry.priority = int(priority)
            entry.source_path = source_path
            entry.kind = kind
            entry.last_touched = _now()
            if note:
                entry.notes.append(note)
            report = self._reports_by_path.get(source_path)
            if report is not None:
                entry.last_method = report.method
                entry.last_report = report.as_dict()
                entry.current_device = report.actual_device

        self.refresh_runtime_state()
        return entry

    def entry_for_object(self, obj: Any) -> ResidencyEntry | None:
        if obj is None:
            return None
        entry_id = getattr(obj, "__gpu_resident_loader_entry_id__", None)
        if entry_id is None:
            return None
        with self._lock:
            return self._entries.get(entry_id)

    def set_sticky(self, obj: Any, sticky: bool, priority: int | None = None) -> ResidencyEntry | None:
        entry = self.entry_for_object(obj)
        if entry is None:
            return None
        with self._lock:
            entry.sticky = bool(sticky)
            if priority is not None:
                entry.priority = int(priority)
            entry.last_touched = _now()
        return entry

    def touch(self, obj: Any) -> None:
        entry = self.entry_for_object(obj)
        if entry is None:
            return
        with self._lock:
            entry.last_touched = _now()

    def sticky_loaded_wrappers(self, device: torch.device | None) -> list[Any]:
        try:
            import comfy.model_management as model_management
        except Exception:
            return []

        output: list[Any] = []
        with self._lock:
            for loaded in list(model_management.current_loaded_models):
                if device is not None and loaded.device != device:
                    continue
                entry = self.entry_for_object(loaded.model)
                if entry is not None and entry.sticky:
                    output.append(loaded)
        return output

    def refresh_runtime_state(self) -> None:
        try:
            import comfy.model_management as model_management
        except Exception:
            return

        with self._lock:
            for entry in self._entries.values():
                if not entry.is_alive():
                    continue
                obj = entry.object()
                if obj is None:
                    continue
                load_device = getattr(obj, "load_device", None)
                offload_device = getattr(obj, "offload_device", None)
                if load_device is not None:
                    entry.load_device = str(load_device)
                if offload_device is not None:
                    entry.offload_device = str(offload_device)

            for loaded in list(model_management.current_loaded_models):
                entry = self.entry_for_object(loaded.model)
                if entry is None:
                    continue
                entry.loaded_bytes = int(loaded.model_loaded_memory())
                entry.total_bytes = int(loaded.model_memory())
                entry.current_device = str(loaded.device)
                entry.last_touched = _now()

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

    def snapshot_json(self) -> str:
        payload = {
            "policy": self.get_policy(),
            "entries": self.snapshot(),
        }
        return json.dumps(payload, indent=2, sort_keys=True)


REGISTRY = ResidencyRegistry()
