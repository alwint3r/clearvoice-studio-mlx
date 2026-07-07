"""Torch checkpoint conversion helpers for MLX models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def torch_key_to_mlx_key(key: str) -> str:
    """Map PyTorch state-dict keys to the MLX module tree."""
    key = key.removeprefix("module.")
    # create_mlx_model returns the same wrapper boundary as the torch model.
    # Its trainable graph is nested under `model`.
    key = f"model.{key}"
    return key


def torch_tensor_to_mlx_value(key: str, value: Any, mlx_key: str | None = None) -> mx.array:
    """Convert a PyTorch tensor into the shape expected by MLX layers."""
    array = _tensor_to_numpy(value)

    if key.removeprefix("module.") == "dec.weight" and array.ndim == 3:
        # Torch ConvTranspose1d: (in, out/groups, kernel).
        # MLX ConvTranspose1d: (out, kernel, in).
        array = np.transpose(array, (1, 2, 0))
    elif key.endswith(".weight") and array.ndim == 3:
        # Torch Conv1d: (out, in, kernel). MLX Conv1d: (out, kernel, in).
        array = np.transpose(array, (0, 2, 1))
    elif key.endswith(".weight") and array.ndim == 4:
        # Torch Conv2d: (out, in/groups, kh, kw). MLX Conv2d: (out, kh, kw, in/groups).
        array = np.transpose(array, (0, 2, 3, 1))

    if mlx_key and array.ndim == 1 and (
        mlx_key.endswith(".mask_net.norm.weight")
        or mlx_key.endswith(".mask_net.norm.bias")
        or mlx_key.endswith(".mossformer.norm.weight")
        or mlx_key.endswith(".mossformer.norm.bias")
        or mlx_key.endswith(".intra_norm.weight")
        or mlx_key.endswith(".intra_norm.bias")
    ):
        array = array[:, None]

    return mx.array(array)


def extract_state_dict(checkpoint: dict[str, Any], model_key: str = "model") -> dict[str, Any]:
    if model_key in checkpoint:
        checkpoint = checkpoint[model_key]
    weights = {}
    for key, value in checkpoint.items():
        mlx_key = torch_key_to_mlx_key(key)
        weights[mlx_key] = torch_tensor_to_mlx_value(key, value, mlx_key=mlx_key)
    return weights


def nest_dotted_keys(weights: dict[str, mx.array]) -> dict[str, Any]:
    """Turn flat dotted keys into the nested dict expected by mlx.nn.Module.update."""
    trie: dict[str, Any] = {}
    for key, value in weights.items():
        current = trie
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    def convert_numeric_dicts(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        converted = {key: convert_numeric_dicts(value) for key, value in node.items()}
        if converted and all(key.isdigit() for key in converted):
            return [converted.get(str(index), {}) for index in range(max(int(key) for key in converted) + 1)]
        return converted

    return convert_numeric_dicts(trie)


def convert_torch_checkpoint(checkpoint_path: str | Path, output_path: str | Path, model_key: str = "model") -> None:
    """Convert a ClearerVoice PyTorch checkpoint to an MLX safetensors file."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    weights = extract_state_dict(checkpoint, model_key=model_key)
    mx.save_safetensors(str(output_path), weights)
