"""MLX layers mirroring ClearerVoice Torch helpers."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def nch_to_nlc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 1))


def nlc_to_nch(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 1))


class GlobalLayerNorm(nn.Module):
    def __init__(self, dim: int, shape: int, eps: float = 1e-8, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.shape = shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            if shape == 3:
                self.weight = mx.ones((dim, 1))
                self.bias = mx.zeros((dim, 1))
            elif shape == 4:
                self.weight = mx.ones((dim, 1, 1))
                self.bias = mx.zeros((dim, 1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        axes = (1, 2) if x.ndim == 3 else (1, 2, 3)
        mean = mx.mean(x, axis=axes, keepdims=True)
        var = mx.mean(mx.square(x - mean), axis=axes, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        if self.elementwise_affine:
            x = self.weight * x + self.bias
        return x


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channel dimension for Torch-style NCL tensors."""

    def __init__(self, dim: int, eps: float = 1e-8, affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, affine=affine)

    def __call__(self, x: mx.array) -> mx.array:
        return nlc_to_nch(self.norm(nch_to_nlc(x)))


class TorchGroupNorm(nn.Module):
    """GroupNorm for Torch-style NCL arrays."""

    def __init__(self, num_groups: int, dim: int, eps: float = 1e-8, affine: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = mx.ones((dim, 1))
            self.bias = mx.zeros((dim, 1))

    def __call__(self, x: mx.array) -> mx.array:
        batch, channels, length = x.shape
        x_grouped = mx.reshape(x, (batch, self.num_groups, channels // self.num_groups, length))
        mean = mx.mean(x_grouped, axis=(2, 3), keepdims=True)
        var = mx.mean(mx.square(x_grouped - mean), axis=(2, 3), keepdims=True)
        x = mx.reshape((x_grouped - mean) / mx.sqrt(var + self.eps), (batch, channels, length))
        if self.affine:
            x = x * self.weight + self.bias
        return x


def select_norm(norm: str, dim: int, shape: int) -> nn.Module:
    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return ChannelLayerNorm(dim)
    if norm == "ln":
        return TorchGroupNorm(1, dim, eps=1e-8)
    return nn.BatchNorm(dim)


class TorchConv1d(nn.Module):
    """Conv1d wrapper that accepts and returns Torch-style NCL arrays."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return nlc_to_nch(self.conv(nch_to_nlc(x)))


class TorchConvTranspose1d(nn.Module):
    """ConvTranspose1d wrapper that accepts and returns Torch-style NCL arrays."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, bias: bool = True):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return nlc_to_nch(self.conv(nch_to_nlc(x)))


class PReLU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.weight = mx.full((num_parameters,), init)

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight
        if weight.size == 1:
            return mx.maximum(x, 0) + weight * mx.minimum(x, 0)
        shape = (1, weight.shape[0]) + (1,) * (x.ndim - 2)
        return mx.maximum(x, 0) + mx.reshape(weight, shape) * mx.minimum(x, 0)
