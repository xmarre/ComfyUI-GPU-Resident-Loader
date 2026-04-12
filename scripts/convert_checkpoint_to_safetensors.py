#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a PyTorch checkpoint into a safetensors state dict file.")
    parser.add_argument("--input", required=True, help="Input .ckpt/.pt/.bin/.pth path")
    parser.add_argument("--output", required=True, help="Output .safetensors path")
    parser.add_argument(
        "--state-dict-key",
        default="state_dict",
        help="Top-level key to extract when the checkpoint is a wrapper object. Default: state_dict",
    )
    parser.add_argument(
        "--allow-non-tensor-values",
        action="store_true",
        help="Ignore non-tensor items instead of failing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and args.state_dict_key in checkpoint and isinstance(checkpoint[args.state_dict_key], dict):
        state_dict = checkpoint[args.state_dict_key]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError("Checkpoint is not a dictionary and no state_dict could be extracted.")

    output_tensors = {}
    skipped = []
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            output_tensors[key] = value.detach().cpu().contiguous()
        elif args.allow_non_tensor_values:
            skipped.append(key)
        else:
            raise TypeError(f"Key {key!r} is not a tensor. Re-run with --allow-non-tensor-values to skip it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "converted_from": os.path.basename(str(input_path)),
        "converter": "ComfyUI-GPU-Resident-Loader",
    }
    save_file(output_tensors, str(output_path), metadata=metadata)

    print(f"Wrote {len(output_tensors)} tensors to {output_path}")
    if skipped:
        print(f"Skipped {len(skipped)} non-tensor keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
