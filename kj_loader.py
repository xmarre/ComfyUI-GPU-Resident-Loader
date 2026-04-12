from __future__ import annotations

import logging
from typing import Any

import folder_paths
import torch

import comfy.sd
import comfy.utils
from comfy.cli_args import PerformanceFeature, args
from comfy.ldm.modules.attention import attention_pytorch, wrap_attn

from .residency import KIND_CHECKPOINT, KIND_MODEL, REGISTRY


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


def _set_cublas_linear(enabled: bool) -> None:
    if enabled:
        args.fast.add(PerformanceFeature.CublasOps)
    else:
        args.fast.discard(PerformanceFeature.CublasOps)


def _set_fp16_accumulation(enabled: bool) -> None:
    flag = getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", None)
    if flag is None:
        raise RuntimeError(
            "Failed to set fp16 accumulation. This requires a PyTorch build exposing "
            "torch.backends.cuda.matmul.allow_fp16_accumulation."
        )
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)


def get_sage_func(sage_attention: str, allow_compile: bool = False):
    _LOG.info("GPU Resident Loader: using sage attention mode %s", sage_attention)
    from sageattention import sageattn

    if sage_attention == "auto":
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
    ):
        _set_cublas_linear(patch_cublaslinear)
        _set_fp16_accumulation(enable_fp16_accumulation)

        model_options = _build_model_options(weight_dtype)
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        explicit_device = REGISTRY.explicit_load_device(kind=KIND_MODEL, source_path=unet_path)

        with REGISTRY.load_context(kind=KIND_MODEL, source_path=unet_path, explicit_device=explicit_device):
            sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
            if extra_state_dict:
                extra_sd = comfy.utils.load_torch_file(extra_state_dict)
                sd.update(extra_sd)
                del extra_sd

        diffusion_model_prefix = comfy.sd.model_detection.unet_prefix_from_state_dict(sd)
        sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=False)
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
        _apply_model_postload_options(model, compute_dtype=compute_dtype, sage_attention=sage_attention)
        REGISTRY.bind_object(model, source_path=unet_path, kind=KIND_MODEL)
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
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"
    CATEGORY = "GPU Resident Loader/loaders"
    DESCRIPTION = (
        "Checkpoint loader with the KJ DiffusionModelLoader-style tuning knobs plus GPU-resident ingest. "
        "It loads the whole checkpoint, then binds the model, CLIP, and VAE into the residency registry."
    )

    def load(
        self,
        ckpt_name: str,
        weight_dtype: str,
        compute_dtype: str,
        patch_cublaslinear: bool,
        sage_attention: str,
        enable_fp16_accumulation: bool,
    ):
        _set_cublas_linear(patch_cublaslinear)
        _set_fp16_accumulation(enable_fp16_accumulation)

        model_options = _build_model_options(weight_dtype)
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        explicit_device = REGISTRY.explicit_load_device(kind=KIND_CHECKPOINT, source_path=ckpt_path)

        with REGISTRY.load_context(kind=KIND_CHECKPOINT, source_path=ckpt_path, explicit_device=explicit_device):
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
        REGISTRY.bind_object(model, source_path=ckpt_path, kind=KIND_MODEL, note="checkpoint model")
        if clip is not None and getattr(clip, "patcher", None) is not None:
            REGISTRY.bind_object(clip.patcher, source_path=ckpt_path, kind=KIND_CLIP, note="checkpoint clip")
        if vae is not None and getattr(vae, "patcher", None) is not None:
            REGISTRY.bind_object(vae.patcher, source_path=ckpt_path, kind=KIND_VAE, note="checkpoint vae")
        return model, clip, vae
