from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any

import folder_paths
import torch

import comfy.sd
import comfy.utils
from comfy.cli_args import PerformanceFeature, args
from comfy.ldm.modules.attention import attention_pytorch, wrap_attn

from .patches import infer_unet_prefix_from_keys, load_safetensors_state_dict
from .residency import KIND_CHECKPOINT, KIND_CLIP, KIND_MODEL, KIND_VAE, POLICIES, REGISTRY


_LOG = logging.getLogger(__name__)

SAGE_ATTN_MODES = [
    "disabled",
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
]

DTYPE_MAP = {
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

UNET_PREFIX_CANDIDATES = (
    "model.diffusion_model.",
    "model.model.",
    "net.",
    "model.",
)


def _set_cublas_linear(enabled: bool) -> None:
    if enabled:
        args.fast.add(PerformanceFeature.CublasOps)
    else:
        args.fast.discard(PerformanceFeature.CublasOps)


def _set_fp16_accumulation(enabled: bool) -> None:
    if not hasattr(torch.backends.cuda, "matmul"):
        if enabled:
            raise RuntimeError(
                "Failed to enable fp16 accumulation. This requires a PyTorch build exposing "
                "torch.backends.cuda.matmul.allow_fp16_accumulation."
            )
        _LOG.warning(
            "GPU Resident Loader: fp16 accumulation toggle is unavailable on this PyTorch build; leaving default behavior."
        )
        return

    flag = getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", None)
    if flag is None:
        if enabled:
            raise RuntimeError(
                "Failed to enable fp16 accumulation. This requires a PyTorch build exposing "
                "torch.backends.cuda.matmul.allow_fp16_accumulation."
            )
        _LOG.warning(
            "GPU Resident Loader: fp16 accumulation toggle is unavailable on this PyTorch build; leaving default behavior."
        )
        return
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)


def _get_fp16_accumulation_state() -> bool | None:
    matmul = getattr(torch.backends.cuda, "matmul", None)
    return getattr(matmul, "allow_fp16_accumulation", None)


@contextlib.contextmanager
def _temporary_backend_flags(*, cublas: bool, fp16_accumulation: bool):
    prev_cublas = PerformanceFeature.CublasOps in args.fast
    prev_fp16 = _get_fp16_accumulation_state()

    try:
        _set_cublas_linear(cublas)
        _set_fp16_accumulation(fp16_accumulation)
        yield
    finally:
        _set_cublas_linear(prev_cublas)
        if prev_fp16 is not None:
            torch.backends.cuda.matmul.allow_fp16_accumulation = prev_fp16


def get_sage_func(sage_attention: str, allow_compile: bool = False):
    _LOG.info("GPU Resident Loader: using sage attention mode %s", sage_attention)

    if sage_attention == "auto":
        from sageattention import sageattn

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
    elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_cuda(
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=attn_mask,
                pv_accum_dtype="fp32",
                tensor_layout=tensor_layout,
            )
    elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp16_triton(
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=attn_mask,
                tensor_layout=tensor_layout,
            )
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=attn_mask,
                pv_accum_dtype="fp32+fp32",
                tensor_layout=tensor_layout,
            )
    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
            return sageattn_qk_int8_pv_fp8_cuda(
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=attn_mask,
                pv_accum_dtype="fp32+fp16",
                tensor_layout=tensor_layout,
            )
    elif "sageattn3" in sage_attention:
        from sageattn3 import sageattn3_blackwell

        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD", **kwargs):
            q, k, v = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
            out = sageattn3_blackwell(
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=attn_mask,
                per_block_mean=(sage_attention == "sageattn3_per_block_mean"),
            )
            return out.transpose(1, 2) if tensor_layout == "NHD" else out
    else:
        raise ValueError(f"Unsupported sage attention mode: {sage_attention}")

    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None)
    if not allow_compile and callable(disable):
        sage_func = disable()(sage_func)

    @wrap_attn
    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if kwargs.get("low_precision_attention", True) is False:
            return attention_pytorch(
                q,
                k,
                v,
                heads,
                mask=mask,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        in_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)

        if skip_reshape:
            batch, _, _, dim_head = q.shape
            tensor_layout = "HND"
        else:
            batch, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = [tensor.view(batch, -1, heads, dim_head) for tensor in (q, k, v)]
            tensor_layout = "NHD"

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout).to(in_dtype)
        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(batch, -1, heads * dim_head)
        else:
            if skip_output_reshape:
                out = out.transpose(1, 2)
            else:
                out = out.reshape(batch, -1, heads * dim_head)
        return out

    return attention_sage


def _apply_model_postload_options(model, *, compute_dtype: str, sage_attention: str) -> None:
    if dtype := DTYPE_MAP.get(compute_dtype):
        model.set_model_compute_dtype(dtype)
        model.force_cast_weights = False
        _LOG.info("GPU Resident Loader: set compute dtype to %s", dtype)

    if sage_attention != "disabled":
        new_attention = get_sage_func(sage_attention)

        def attention_override_sage(func, *args, **kwargs):
            return new_attention.__wrapped__(*args, **kwargs)

        model.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = attention_override_sage


def _build_model_options(weight_dtype: str) -> dict[str, Any]:
    model_options: dict[str, Any] = {}
    if dtype := DTYPE_MAP.get(weight_dtype):
        model_options["dtype"] = dtype
        _LOG.info("GPU Resident Loader: set weight dtype to %s", dtype)
    if weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    return model_options


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _effective_policy_name(policy_override: str | None) -> str:
    return REGISTRY.get_policy() if policy_override is None else policy_override


def _make_loader_key(loader_name: str, **payload: Any) -> str:
    normalized_payload = {"loader": loader_name}
    normalized_payload.update(payload)
    return json.dumps(normalized_payload, sort_keys=True, separators=(",", ":"))


def _normalize_optional_policy(value: str | None) -> str | None:
    normalized = _normalize_optional_string(value)
    return None if normalized is None else normalized.lower()


def _resolve_loader_policy_and_extra_state_dict(
    *,
    loader_name: str,
    policy_override: str | None,
    extra_state_dict: str | None,
) -> tuple[str | None, str | None]:
    normalized_policy = _normalize_optional_policy(policy_override)
    if normalized_policy is not None and normalized_policy not in POLICIES:
        raise ValueError(
            f"{loader_name}: unsupported policy_override {normalized_policy!r}. "
            f"Expected one of: {', '.join(POLICIES)}."
        )

    normalized_extra = _normalize_optional_string(extra_state_dict)
    if normalized_extra is None:
        return normalized_policy, None

    if normalized_policy is None and normalized_extra.lower() in POLICIES and not os.path.exists(normalized_extra):
        _LOG.warning(
            "GPU Resident Loader: interpreting legacy extra_state_dict value %r as policy_override for %s",
            normalized_extra,
            loader_name,
        )
        return normalized_extra.lower(), None

    if not os.path.isfile(normalized_extra):
        raise FileNotFoundError(
            f"{loader_name}: extra_state_dict must point to an existing state-dict file, got {normalized_extra!r}. "
            f"If you meant to pass a residency policy, connect that STRING to policy_override instead."
        )

    return normalized_policy, normalized_extra


def _is_safetensors_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith(".safetensors") or lowered.endswith(".sft")


def _warn_pickle_checkpoint_gpu_compatibility(loader_name: str, path: str) -> None:
    if _is_safetensors_path(path):
        return
    if not REGISTRY.wants_gpu_ingest(KIND_MODEL):
        return
    _LOG.warning(
        "%s: %s is using the compatibility path through torch.load() on CPU before tensor-by-tensor copies to GPU. "
        "Convert hot checkpoints to safetensors with scripts/convert_checkpoint_to_safetensors.py for the real fast path.",
        loader_name,
        path,
    )


def _selected_unet_key_map_from_header(
    path: str,
    *,
    known_unet_keys: set[str] | None = None,
) -> tuple[dict[str, str], str | None]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        all_keys = list(handle.keys())

    prefix = infer_unet_prefix_from_keys(all_keys)
    if known_unet_keys is None:
        selected = {key: key[len(prefix):] for key in all_keys if key.startswith(prefix)}
        return (selected or {key: key for key in all_keys}), prefix

    prefixes_to_try: list[str] = []
    for candidate in (prefix, *UNET_PREFIX_CANDIDATES):
        if candidate and candidate not in prefixes_to_try:
            prefixes_to_try.append(candidate)

    selected: dict[str, str] = {}
    for key in all_keys:
        if key in known_unet_keys:
            selected[key] = key
            continue
        for candidate in prefixes_to_try:
            if key.startswith(candidate):
                stripped = key[len(candidate):]
                if stripped in known_unet_keys:
                    selected[key] = stripped
                    break
    return selected, prefix


def _load_matching_extra_unet_state_dict(
    extra_state_dict_path: str,
    *,
    requested_device: torch.device,
    known_unet_keys: set[str],
) -> dict[str, Any]:
    if _is_safetensors_path(extra_state_dict_path):
        selected_keys, _ = _selected_unet_key_map_from_header(
            extra_state_dict_path,
            known_unet_keys=known_unet_keys,
        )
        extra_sd, _, _, _ = load_safetensors_state_dict(
            extra_state_dict_path,
            requested_device,
            selected_keys=selected_keys,
        )
        return extra_sd

    extra_sd = comfy.utils.load_torch_file(extra_state_dict_path)
    return _extract_unet_state_dict(extra_sd, known_unet_keys=known_unet_keys)


def _extract_unet_state_dict(
    sd: dict[str, Any],
    *,
    diffusion_model_prefix: str | None = None,
    known_unet_keys: set[str] | None = None,
) -> dict[str, Any]:
    if known_unet_keys is None:
        if diffusion_model_prefix is None:
            diffusion_model_prefix = comfy.sd.model_detection.unet_prefix_from_state_dict(sd)
        if diffusion_model_prefix:
            prefix_len = len(diffusion_model_prefix)
            extracted = {key[prefix_len:]: value for key, value in sd.items() if key.startswith(diffusion_model_prefix)}
            if extracted:
                return extracted
        return dict(sd)

    prefixes_to_try: list[str] = []

    def add_prefix(prefix: str | None) -> None:
        if prefix and prefix not in prefixes_to_try:
            prefixes_to_try.append(prefix)

    add_prefix(diffusion_model_prefix)
    add_prefix(comfy.sd.model_detection.unet_prefix_from_state_dict(sd))
    for prefix in UNET_PREFIX_CANDIDATES:
        add_prefix(prefix)

    extracted: dict[str, Any] = {}
    for key, value in sd.items():
        if key in known_unet_keys:
            extracted[key] = value
            continue
        for prefix in prefixes_to_try:
            if not key.startswith(prefix):
                continue
            stripped = key[len(prefix):]
            if stripped in known_unet_keys:
                extracted[stripped] = value
                break
    return extracted


@contextlib.contextmanager
def _apply_policy_override(policy_override: str | None):
    if policy_override is None:
        yield
        return

    previous_policy = REGISTRY.get_policy()
    if previous_policy == policy_override:
        yield
        return

    REGISTRY.set_policy(policy_override)
    try:
        yield
    finally:
        REGISTRY.set_policy(previous_policy)


def _bind_model_for_reuse(model, *, source_path: str, note: str, loader_key: str) -> None:
    REGISTRY.bind_object(model, source_path=source_path, kind=KIND_MODEL, note=note, loader_key=loader_key)


def _bind_clip_for_reuse(clip, *, source_path: str, note: str, loader_key: str) -> None:
    if clip is not None and getattr(clip, "patcher", None) is not None:
        REGISTRY.bind_object(
            clip.patcher,
            source_path=source_path,
            kind=KIND_CLIP,
            note=note,
            loader_key=loader_key,
            reusable_obj=clip,
        )


def _bind_vae_for_reuse(vae, *, source_path: str, note: str, loader_key: str) -> None:
    if vae is not None and getattr(vae, "patcher", None) is not None:
        REGISTRY.bind_object(
            vae.patcher,
            source_path=source_path,
            kind=KIND_VAE,
            note=note,
            loader_key=loader_key,
            reusable_obj=vae,
        )


def _load_resident_diffusion_model(
    *,
    loader_name: str,
    cache_scope: str,
    source_path: str,
    note: str,
    weight_dtype: str,
    compute_dtype: str,
    patch_cublaslinear: bool,
    sage_attention: str,
    enable_fp16_accumulation: bool,
    extra_state_dict: str | None = None,
    policy_override: str | None = None,
) -> Any:
    model_options = _build_model_options(weight_dtype)
    effective_policy = _effective_policy_name(policy_override)
    loader_key = _make_loader_key(
        cache_scope,
        component="model",
        weight_dtype=weight_dtype,
        compute_dtype=compute_dtype,
        patch_cublaslinear=patch_cublaslinear,
        sage_attention=sage_attention,
        enable_fp16_accumulation=enable_fp16_accumulation,
        extra_state_dict=extra_state_dict,
        policy=effective_policy,
    )
    reused_model = REGISTRY.lookup_live_object(kind=KIND_MODEL, source_path=source_path, loader_key=loader_key)
    if reused_model is not None:
        _LOG.info("%s: reusing live model for %s", loader_name, source_path)
        return reused_model

    _warn_pickle_checkpoint_gpu_compatibility(loader_name, source_path)
    explicit_device = REGISTRY.explicit_load_device(kind=KIND_MODEL, source_path=source_path)

    with _temporary_backend_flags(
        cublas=patch_cublaslinear,
        fp16_accumulation=enable_fp16_accumulation,
    ):
        with REGISTRY.load_context(
            kind=KIND_MODEL,
            source_path=source_path,
            explicit_device=explicit_device,
            cache_key=loader_key,
        ):
            sd, metadata = comfy.utils.load_torch_file(source_path, return_metadata=True)
            if not _is_safetensors_path(source_path):
                sd = _extract_unet_state_dict(sd)
            if extra_state_dict:
                requested_device = explicit_device if explicit_device is not None else torch.device("cpu")
                sd.update(
                    _load_matching_extra_unet_state_dict(
                        extra_state_dict,
                        requested_device=requested_device,
                        known_unet_keys=set(sd),
                    )
                )

        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
        _apply_model_postload_options(model, compute_dtype=compute_dtype, sage_attention=sage_attention)

    _bind_model_for_reuse(model, source_path=source_path, note=note, loader_key=loader_key)
    return model


def _checkpoint_model_loader_key(
    *,
    weight_dtype: str,
    compute_dtype: str,
    patch_cublaslinear: bool,
    sage_attention: str,
    enable_fp16_accumulation: bool,
    policy_override: str | None,
) -> str:
    return _make_loader_key(
        "checkpoint_model",
        component="model",
        weight_dtype=weight_dtype,
        compute_dtype=compute_dtype,
        patch_cublaslinear=patch_cublaslinear,
        sage_attention=sage_attention,
        enable_fp16_accumulation=enable_fp16_accumulation,
        extra_state_dict=None,
        policy=_effective_policy_name(policy_override),
    )


def _checkpoint_component_loader_key(component: str, policy_override: str | None) -> str:
    return _make_loader_key(f"checkpoint_{component}", component=component, policy=_effective_policy_name(policy_override))


def _load_checkpoint_clip_only(
    *,
    ckpt_path: str,
    policy_override: str | None,
    loader_name: str,
):
    loader_key = _checkpoint_component_loader_key("clip", policy_override)
    reused_clip = REGISTRY.lookup_live_object(kind=KIND_CLIP, source_path=ckpt_path, loader_key=loader_key)
    if reused_clip is not None:
        _LOG.info("%s: reusing live CLIP for %s", loader_name, ckpt_path)
        return reused_clip

    _warn_pickle_checkpoint_gpu_compatibility(loader_name, ckpt_path)
    explicit_device = REGISTRY.explicit_load_device(kind=KIND_CLIP, source_path=ckpt_path)
    with REGISTRY.load_context(kind=KIND_CLIP, source_path=ckpt_path, explicit_device=explicit_device, cache_key=loader_key):
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
    _, clip, _, _ = comfy.sd.load_state_dict_guess_config(
        sd,
        output_vae=False,
        output_clip=True,
        output_model=False,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        metadata=metadata,
    )
    _bind_clip_for_reuse(clip, source_path=ckpt_path, note="checkpoint clip", loader_key=loader_key)
    return clip


def _load_checkpoint_vae_only(
    *,
    ckpt_path: str,
    policy_override: str | None,
    loader_name: str,
):
    loader_key = _checkpoint_component_loader_key("vae", policy_override)
    reused_vae = REGISTRY.lookup_live_object(kind=KIND_VAE, source_path=ckpt_path, loader_key=loader_key)
    if reused_vae is not None:
        _LOG.info("%s: reusing live VAE for %s", loader_name, ckpt_path)
        return reused_vae

    _warn_pickle_checkpoint_gpu_compatibility(loader_name, ckpt_path)
    explicit_device = REGISTRY.explicit_load_device(kind=KIND_VAE, source_path=ckpt_path)
    with REGISTRY.load_context(kind=KIND_VAE, source_path=ckpt_path, explicit_device=explicit_device, cache_key=loader_key):
        sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
    _, _, vae, _ = comfy.sd.load_state_dict_guess_config(
        sd,
        output_vae=True,
        output_clip=False,
        output_model=False,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        metadata=metadata,
    )
    _bind_vae_for_reuse(vae, source_path=ckpt_path, note="checkpoint vae", loader_key=loader_key)
    return vae


def _load_full_checkpoint(
    *,
    ckpt_path: str,
    weight_dtype: str,
    compute_dtype: str,
    patch_cublaslinear: bool,
    sage_attention: str,
    enable_fp16_accumulation: bool,
    policy_override: str | None,
    loader_name: str,
):
    model_key = _checkpoint_model_loader_key(
        weight_dtype=weight_dtype,
        compute_dtype=compute_dtype,
        patch_cublaslinear=patch_cublaslinear,
        sage_attention=sage_attention,
        enable_fp16_accumulation=enable_fp16_accumulation,
        policy_override=policy_override,
    )
    clip_key = _checkpoint_component_loader_key("clip", policy_override)
    vae_key = _checkpoint_component_loader_key("vae", policy_override)

    model = REGISTRY.lookup_live_object(kind=KIND_MODEL, source_path=ckpt_path, loader_key=model_key)
    clip = REGISTRY.lookup_live_object(kind=KIND_CLIP, source_path=ckpt_path, loader_key=clip_key)
    vae = REGISTRY.lookup_live_object(kind=KIND_VAE, source_path=ckpt_path, loader_key=vae_key)
    missing = [name for name, value in (("model", model), ("clip", clip), ("vae", vae)) if value is None]
    if not missing:
        _LOG.info("%s: reusing live checkpoint outputs for %s", loader_name, ckpt_path)
        return model, clip, vae

    if len(missing) == 1:
        if missing[0] == "model":
            model = _load_resident_diffusion_model(
                loader_name=loader_name,
                cache_scope="checkpoint_model",
                source_path=ckpt_path,
                note="checkpoint model",
                weight_dtype=weight_dtype,
                compute_dtype=compute_dtype,
                patch_cublaslinear=patch_cublaslinear,
                sage_attention=sage_attention,
                enable_fp16_accumulation=enable_fp16_accumulation,
                policy_override=policy_override,
            )
        elif missing[0] == "clip":
            clip = _load_checkpoint_clip_only(
                ckpt_path=ckpt_path,
                policy_override=policy_override,
                loader_name=loader_name,
            )
        else:
            vae = _load_checkpoint_vae_only(
                ckpt_path=ckpt_path,
                policy_override=policy_override,
                loader_name=loader_name,
            )
        return model, clip, vae

    _warn_pickle_checkpoint_gpu_compatibility(loader_name, ckpt_path)
    model_options = _build_model_options(weight_dtype)
    explicit_device = REGISTRY.explicit_load_device(kind=KIND_CHECKPOINT, source_path=ckpt_path)
    with _temporary_backend_flags(
        cublas=patch_cublaslinear,
        fp16_accumulation=enable_fp16_accumulation,
    ):
        with REGISTRY.load_context(
            kind=KIND_CHECKPOINT,
            source_path=ckpt_path,
            explicit_device=explicit_device,
            cache_key=_make_loader_key(
                loader_name,
                component="full_checkpoint",
                weight_dtype=weight_dtype,
                compute_dtype=compute_dtype,
                patch_cublaslinear=patch_cublaslinear,
                sage_attention=sage_attention,
                enable_fp16_accumulation=enable_fp16_accumulation,
                policy=_effective_policy_name(policy_override),
            ),
        ):
            sd, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
        model, clip, vae, _ = comfy.sd.load_state_dict_guess_config(
            sd,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            metadata=metadata,
            model_options=model_options,
        )
        _apply_model_postload_options(model, compute_dtype=compute_dtype, sage_attention=sage_attention)

    _bind_model_for_reuse(model, source_path=ckpt_path, note="checkpoint model", loader_key=model_key)
    _bind_clip_for_reuse(clip, source_path=ckpt_path, note="checkpoint clip", loader_key=clip_key)
    _bind_vae_for_reuse(vae, source_path=ckpt_path, note="checkpoint vae", loader_key=vae_key)
    return model, clip, vae


class DiffusionModelSelectorResident:
    @classmethod
    def INPUT_TYPES(cls):
        ltx2_connector_models = folder_paths.get_filename_list("text_encoders")
        ltx2_connector_models = [m for m in ltx2_connector_models if "connector" in m.lower()]
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models") + ltx2_connector_models,
                    {"tooltip": "The name of the diffusion model or connector to resolve."},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "get_path"
    CATEGORY = "GPU Resident Loader/loaders"
    DESCRIPTION = "Returns the absolute model path as a string. Mirrors the selector behavior of KJ's diffusion model selector."

    def get_path(self, model_name: str):
        if "connector" in model_name.lower():
            model_path = folder_paths.get_full_path_or_raise("text_encoders", model_name)
        else:
            model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        return (model_path,)


class DiffusionModelLoaderResident:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "The diffusion model file to load."},
                ),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"],),
                "compute_dtype": (
                    ["default", "fp16", "bf16", "fp32"],
                    {"default": "default", "tooltip": "Compute dtype to apply after model creation."},
                ),
                "patch_cublaslinear": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Toggle ComfyUI's cublas_ops performance feature."},
                ),
                "sage_attention": (
                    SAGE_ATTN_MODES,
                    {"default": "disabled", "tooltip": "Patch optimized attention override to a SageAttention variant."},
                ),
                "enable_fp16_accumulation": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Set torch.backends.cuda.matmul.allow_fp16_accumulation."},
                ),
            },
            "optional": {
                "extra_state_dict": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Optional absolute path to a second state dict merged into the main diffusion state dict before model detection.",
                    },
                ),
                "policy_override": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Optional residency policy override. Connect Set Global Residency Policy here, not to extra_state_dict.",
                    },
                )
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_and_load"
    CATEGORY = "GPU Resident Loader/loaders"
    DESCRIPTION = (
        "KJ-compatible diffusion-model loader with GPU-resident ingest and residency tracking. "
        "It mirrors the KJ node's weight dtype, compute dtype, cublas_ops, SageAttention, fp16 accumulation, "
        "and optional extra-state-dict features."
    )

    def patch_and_load(
        self,
        model_name: str,
        weight_dtype: str,
        compute_dtype: str,
        patch_cublaslinear: bool,
        sage_attention: str,
        enable_fp16_accumulation: bool,
        extra_state_dict: str | None = None,
        policy_override: str | None = None,
    ):
        policy_override, extra_state_dict = _resolve_loader_policy_and_extra_state_dict(
            loader_name="Diffusion Model Loader Resident",
            policy_override=policy_override,
            extra_state_dict=extra_state_dict,
        )

        with _apply_policy_override(policy_override):
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
            model = _load_resident_diffusion_model(
                loader_name="Diffusion Model Loader Resident",
                cache_scope="diffusion_model",
                source_path=unet_path,
                note="diffusion model",
                weight_dtype=weight_dtype,
                compute_dtype=compute_dtype,
                patch_cublaslinear=patch_cublaslinear,
                sage_attention=sage_attention,
                enable_fp16_accumulation=enable_fp16_accumulation,
                extra_state_dict=extra_state_dict,
                policy_override=policy_override,
            )
            return (model,)


class CheckpointLoaderResident:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "Checkpoint file to load."},
                ),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"],),
                "compute_dtype": (
                    ["default", "fp16", "bf16", "fp32"],
                    {"default": "default", "tooltip": "Compute dtype to apply to the diffusion model patcher after load."},
                ),
                "patch_cublaslinear": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Toggle ComfyUI's cublas_ops performance feature."},
                ),
                "sage_attention": (
                    SAGE_ATTN_MODES,
                    {"default": "disabled", "tooltip": "Patch optimized attention override on the loaded model."},
                ),
                "enable_fp16_accumulation": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Set torch.backends.cuda.matmul.allow_fp16_accumulation."},
                ),
            },
            "optional": {
                "policy_override": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Optional residency policy override. Connect Set Global Residency Policy here.",
                    },
                )
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"
    CATEGORY = "GPU Resident Loader/loaders"
    DESCRIPTION = (
        "Checkpoint loader with the KJ DiffusionModelLoader-style tuning knobs plus GPU-resident ingest. "
        "It reuses live checkpoint components when possible and only falls back to a full checkpoint load when multiple outputs are missing."
    )

    def load(
        self,
        ckpt_name: str,
        weight_dtype: str,
        compute_dtype: str,
        patch_cublaslinear: bool,
        sage_attention: str,
        enable_fp16_accumulation: bool,
        policy_override: str | None = None,
    ):
        policy_override, _ = _resolve_loader_policy_and_extra_state_dict(
            loader_name="Checkpoint Loader Resident",
            policy_override=policy_override,
            extra_state_dict=None,
        )

        with _apply_policy_override(policy_override):
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            model, clip, vae = _load_full_checkpoint(
                ckpt_path=ckpt_path,
                weight_dtype=weight_dtype,
                compute_dtype=compute_dtype,
                patch_cublaslinear=patch_cublaslinear,
                sage_attention=sage_attention,
                enable_fp16_accumulation=enable_fp16_accumulation,
                policy_override=policy_override,
                loader_name="Checkpoint Loader Resident",
            )
            return model, clip, vae


class CheckpointModelLoaderResident:
    @classmethod
    def INPUT_TYPES(cls):
        return CheckpointLoaderResident.INPUT_TYPES()

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "GPU Resident Loader/loaders"
    DESCRIPTION = (
        "Checkpoint model-only loader. Safetensors checkpoints take the selective UNet fast path, and live equivalent models are reused when available."
    )

    def load(
        self,
        ckpt_name: str,
        weight_dtype: str,
        compute_dtype: str,
        patch_cublaslinear: bool,
        sage_attention: str,
        enable_fp16_accumulation: bool,
        policy_override: str | None = None,
    ):
        policy_override, _ = _resolve_loader_policy_and_extra_state_dict(
            loader_name="Checkpoint Model Loader Resident",
            policy_override=policy_override,
            extra_state_dict=None,
        )

        with _apply_policy_override(policy_override):
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            model = _load_resident_diffusion_model(
                loader_name="Checkpoint Model Loader Resident",
                cache_scope="checkpoint_model",
                source_path=ckpt_path,
                note="checkpoint model",
                weight_dtype=weight_dtype,
                compute_dtype=compute_dtype,
                patch_cublaslinear=patch_cublaslinear,
                sage_attention=sage_attention,
                enable_fp16_accumulation=enable_fp16_accumulation,
                policy_override=policy_override,
            )
            return (model,)


class CheckpointClipLoaderResident:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "Checkpoint file to load."},
                ),
            },
            "optional": {
                "policy_override": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Optional residency policy override. Connect Set Global Residency Policy here.",
                    },
                )
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "GPU Resident Loader/loaders"
    DESCRIPTION = "Checkpoint CLIP-only loader with live-object reuse. It avoids rebuilding the text encoder when an equivalent CLIP object is already alive."

    def load(self, ckpt_name: str, policy_override: str | None = None):
        policy_override, _ = _resolve_loader_policy_and_extra_state_dict(
            loader_name="Checkpoint Clip Loader Resident",
            policy_override=policy_override,
            extra_state_dict=None,
        )

        with _apply_policy_override(policy_override):
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            clip = _load_checkpoint_clip_only(
                ckpt_path=ckpt_path,
                policy_override=policy_override,
                loader_name="Checkpoint Clip Loader Resident",
            )
            return (clip,)


class CheckpointVAELoaderResident:
    @classmethod
    def INPUT_TYPES(cls):
        return CheckpointClipLoaderResident.INPUT_TYPES()

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "GPU Resident Loader/loaders"
    DESCRIPTION = "Checkpoint VAE-only loader with live-object reuse. It avoids rebuilding the VAE when an equivalent object is still alive."

    def load(self, ckpt_name: str, policy_override: str | None = None):
        policy_override, _ = _resolve_loader_policy_and_extra_state_dict(
            loader_name="Checkpoint VAE Loader Resident",
            policy_override=policy_override,
            extra_state_dict=None,
        )

        with _apply_policy_override(policy_override):
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            vae = _load_checkpoint_vae_only(
                ckpt_path=ckpt_path,
                policy_override=policy_override,
                loader_name="Checkpoint VAE Loader Resident",
            )
            return (vae,)
