"""MLX port of the MossFormer2 48 kHz speech-enhancement mask model."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .mossformer2_ss import (
    FLASHShareAFFConvM,
    RotaryEmbedding,
    ScaleNorm,
    CLayerNorm,
    PReLU,
    ScaledSinuEmbedding,
    UniDeepFsmn,
    conv1d_ncl,
    ncl_to_nlc,
    nlc_to_ncl,
)
from ..layers import select_norm


class GatedFSMN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, lorder: int, hidden_size: int):
        super().__init__()
        from .mossformer2_ss import FFConvM

        self.to_u = FFConvM(in_channels, hidden_size, norm_klass=nn.LayerNorm, dropout=0.1)
        self.to_v = FFConvM(in_channels, hidden_size, norm_klass=nn.LayerNorm, dropout=0.1)
        self.fsmn = UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x_u = self.fsmn(self.to_u(x))
        x_v = self.to_v(x)
        return x_v * x_u + residual


class GatedFSMNBlock(nn.Module):
    def __init__(self, dim: int, inner_channels: int = 256):
        super().__init__()
        self.conv1 = [nn.Conv1d(dim, inner_channels, kernel_size=1), PReLU()]
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_fsmn = GatedFSMN(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        conv1 = self.conv1[1](conv1d_ncl(self.conv1[0], nlc_to_ncl(x)))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(ncl_to_nlc(norm1))
        norm2 = self.norm2(nlc_to_ncl(seq_out))
        conv2 = conv1d_ncl(self.conv2, norm2)
        return ncl_to_nlc(conv2) + x


class MossformerBlockGFSMN(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        group_size: int = 256,
        query_key_dim: int = 128,
        expansion_factor: float = 4.0,
        causal: bool = False,
        attn_dropout: float = 0.1,
        norm_type: str = "scalenorm",
        shift_tokens: bool = True,
    ):
        super().__init__()
        norm_klass = ScaleNorm if norm_type == "scalenorm" else nn.LayerNorm
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))
        self.fsmn = [GatedFSMNBlock(dim) for _ in range(depth)]
        self.layers = [
            FLASHShareAFFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens,
            )
            for _ in range(depth)
        ]

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for index, flash in enumerate(self.layers):
            x = flash(x, mask=mask)
            x = self.fsmn[index](x)
        return x


class MossFormerM(nn.Module):
    def __init__(self, num_blocks: int, d_model: int, causal: bool = False):
        super().__init__()
        self.mossformerM = MossformerBlockGFSMN(
            dim=d_model,
            depth=num_blocks,
            group_size=256,
            query_key_dim=128,
            expansion_factor=4.0,
            causal=causal,
            attn_dropout=0.1,
        )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(self.mossformerM(x))


class ComputationBlock(nn.Module):
    def __init__(self, num_blocks: int, out_channels: int, norm: str = "ln", skip_around_intra: bool = True):
        super().__init__()
        self.intra_mdl = MossFormerM(num_blocks=num_blocks, d_model=out_channels)
        self.skip_around_intra = skip_around_intra
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)

    def __call__(self, x: mx.array) -> mx.array:
        intra = ncl_to_nlc(x)
        intra = self.intra_mdl(intra)
        intra = nlc_to_ncl(intra)
        if self.norm is not None:
            intra = self.intra_norm(intra)
        if self.skip_around_intra:
            intra = intra + x
        return intra


class MossFormerMaskNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        out_channels_final: int,
        num_blocks: int = 24,
        norm: str = "ln",
        num_spks: int = 2,
        skip_around_intra: bool = True,
        use_global_pos_enc: bool = True,
        max_length: int = 20000,
    ):
        super().__init__()
        self.num_spks = num_spks
        self.num_blocks = num_blocks
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc
        if use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)
        self.mdl = ComputationBlock(num_blocks, out_channels, norm, skip_around_intra=skip_around_intra)
        self.conv1d_out = nn.Conv1d(out_channels, out_channels * num_spks, kernel_size=1)
        self.conv1_decoder = nn.Conv1d(out_channels, out_channels_final, 1, bias=False)
        self.prelu = PReLU()
        self.output = [nn.Conv1d(out_channels, out_channels, 1), None]
        self.output_gate = [nn.Conv1d(out_channels, out_channels, 1), None]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = conv1d_ncl(self.conv1d_encoder, x)
        if self.use_global_pos_enc:
            x = x + mx.transpose(self.pos_enc(ncl_to_nlc(x)), (1, 0))[None, :, :]

        x = self.mdl(x)
        x = self.prelu(x)
        x = conv1d_ncl(self.conv1d_out, x)
        batch, _, frames = x.shape
        x = mx.reshape(x, (batch * self.num_spks, -1, frames))
        x = mx.tanh(conv1d_ncl(self.output[0], x)) * mx.sigmoid(conv1d_ncl(self.output_gate[0], x))
        x = conv1d_ncl(self.conv1_decoder, x)
        _, channels, length = x.shape
        x = mx.reshape(x, (batch, self.num_spks, channels, length))
        x = nn.relu(x)
        return mx.transpose(x, (1, 0, 3, 2))[0]


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mossformer = MossFormerMaskNet(in_channels=180, out_channels=512, out_channels_final=961)

    def __call__(self, inputs: mx.array) -> list[mx.array]:
        return [self.mossformer(nlc_to_ncl(inputs))]


class MossFormer2SE48K(nn.Module):
    backend = "mlx"

    def __init__(self, args):
        super().__init__()
        self.model = TestNet()

    def __call__(self, inputs: mx.array) -> list[mx.array]:
        return self.model(inputs)
