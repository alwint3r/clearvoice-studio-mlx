"""MLX port of the MossFormer2 speech-separation inference graph."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..layers import TorchGroupNorm, select_norm


def ncl_to_nlc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 1))


def nlc_to_ncl(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 1))


def nchw_to_nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def nhwc_to_nchw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


def conv1d_ncl(conv: nn.Conv1d, x: mx.array) -> mx.array:
    return nlc_to_ncl(conv(ncl_to_nlc(x)))


def conv_transpose1d_ncl(conv: nn.ConvTranspose1d, x: mx.array) -> mx.array:
    return nlc_to_ncl(conv(ncl_to_nlc(x)))


def conv2d_nchw(conv: nn.Conv2d, x: mx.array) -> mx.array:
    return nhwc_to_nchw(conv(nchw_to_nhwc(x)))


def pad_last_dim(x: mx.array, left: int, right: int, value: float = 0.0) -> mx.array:
    pads = [(0, 0)] * x.ndim
    pads[-1] = (left, right)
    return mx.pad(x, pads, constant_values=value)


def pad_time_dim_2d(x: mx.array, top: int, bottom: int, value: float = 0.0) -> mx.array:
    pads = [(0, 0)] * x.ndim
    pads[-2] = (top, bottom)
    return mx.pad(x, pads, constant_values=value)


def padding_to_multiple_of(n: int, mult: int) -> int:
    remainder = n % mult
    return 0 if remainder == 0 else mult - remainder


class Dropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def __call__(self, x: mx.array) -> mx.array:
        return x


class PReLU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.weight = mx.full((num_parameters,), init)

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight
        if weight.size == 1:
            return mx.maximum(x, 0) + weight * mx.minimum(x, 0)
        if x.ndim == 3:
            shape = (1, weight.shape[0], 1)
        elif x.ndim == 4:
            shape = (1, weight.shape[0], 1, 1)
        else:
            shape = (1,) * (x.ndim - 1) + (weight.shape[0],)
        return mx.maximum(x, 0) + mx.reshape(weight, shape) * mx.minimum(x, 0)


class ScaleNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = mx.ones((1,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.linalg.norm(x, axis=-1, keepdims=True) * self.scale
        return x / mx.maximum(norm, self.eps) * self.g


class CLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))
        self.eps = 1e-5

    def __call__(self, x: mx.array) -> mx.array:
        y = ncl_to_nlc(x)
        mean = mx.mean(y, axis=-1, keepdims=True)
        var = mx.mean(mx.square(y - mean), axis=-1, keepdims=True)
        y = (y - mean) / mx.sqrt(var + self.eps)
        y = y * self.weight + self.bias
        return nlc_to_ncl(y)


class InstanceNorm2d(nn.Module):
    def __init__(self, channels: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = mx.ones((channels,))
            self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=(2, 3), keepdims=True)
        var = mx.mean(mx.square(x - mean), axis=(2, 3), keepdims=True)
        y = (x - mean) / mx.sqrt(var + self.eps)
        if self.affine:
            y = y * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return y


class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = mx.ones((1,))
        self.inv_freq = 1.0 / (10000 ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

    def __call__(self, x: mx.array) -> mx.array:
        steps = mx.arange(x.shape[1]).astype(self.inv_freq.dtype)
        sinu = mx.einsum("i,j->ij", steps, self.inv_freq)
        emb = mx.concatenate([mx.sin(sinu), mx.cos(sinu)], axis=-1)
        return emb * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.freqs = 1.0 / (10000 ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

    def rotate_queries_or_keys(self, x: mx.array) -> mx.array:
        seq_len = x.shape[-2]
        steps = mx.arange(seq_len).astype(self.freqs.dtype)
        freqs = mx.einsum("i,j->ij", steps, self.freqs)
        freqs = mx.reshape(mx.stack([freqs, freqs], axis=-1), (seq_len, -1))
        rot_dim = freqs.shape[-1]
        x_left = x[..., :0]
        x_mid = x[..., :rot_dim]
        x_right = x[..., rot_dim:]
        x_pairs = mx.reshape(x_mid, x_mid.shape[:-1] + (rot_dim // 2, 2))
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]
        x_rot = mx.reshape(mx.stack([-x2, x1], axis=-1), x_mid.shape)
        freqs = freqs.astype(x.dtype)
        x_transformed = x_mid * mx.cos(freqs) + x_rot * mx.sin(freqs)
        return mx.concatenate([x_left, x_transformed, x_right], axis=-1)


class OffsetScale(nn.Module):
    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.gamma = mx.ones((heads, dim))
        self.beta = mx.zeros((heads, dim))

    def __call__(self, x: mx.array) -> list[mx.array]:
        out = mx.einsum("...d,hd->...hd", x, self.gamma) + self.beta
        return [out[..., index, :] for index in range(out.shape[-2])]


class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return conv1d_ncl(self.conv, x)


class ConvModule(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 17):
        super().__init__()
        self.sequential = [
            None,
            DepthwiseConv1d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        return x + ncl_to_nlc(self.sequential[1](nlc_to_ncl(x)))


class FFConvM(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, norm_klass=nn.LayerNorm, dropout: float = 0.1):
        super().__init__()
        self.mdl = [
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            None,
            ConvModule(dim_out),
            Dropout(dropout),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.mdl[0](x)
        x = self.mdl[1](x)
        x = nn.silu(x)
        x = self.mdl[3](x)
        return self.mdl[4](x)


class DilatedDenseNet(nn.Module):
    def __init__(self, depth: int = 4, lorder: int = 20, in_channels: int = 64):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.twidth = lorder * 2 - 1
        self.kernel_size = (self.twidth, 1)
        for i in range(depth):
            dil = 2 ** i
            pad_length = lorder + (dil - 1) * (lorder - 1) - 1
            setattr(self, f"pad{i + 1}", pad_length)
            setattr(
                self,
                f"conv{i + 1}",
                nn.Conv2d(
                    in_channels * (i + 1),
                    in_channels,
                    self.kernel_size,
                    dilation=(dil, 1),
                    groups=in_channels,
                    bias=False,
                ),
            )
            setattr(self, f"norm{i + 1}", InstanceNorm2d(in_channels, affine=True))
            setattr(self, f"prelu{i + 1}", PReLU(in_channels))

    def __call__(self, x: mx.array) -> mx.array:
        skip = x
        for i in range(self.depth):
            pad_length = getattr(self, f"pad{i + 1}")
            out = pad_time_dim_2d(skip, pad_length, pad_length)
            out = conv2d_nchw(getattr(self, f"conv{i + 1}"), out)
            out = getattr(self, f"norm{i + 1}")(out)
            out = getattr(self, f"prelu{i + 1}")(out)
            skip = mx.concatenate([out, skip], axis=1)
        return out


class UniDeepFsmn(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, lorder: int, hidden_size: int):
        super().__init__()
        self.lorder = lorder
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(output_dim, output_dim, (lorder + lorder - 1, 1), groups=output_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        p1 = self.project(nn.relu(self.linear(x)))
        x_per = mx.transpose(mx.expand_dims(p1, axis=1), (0, 3, 2, 1))
        y = pad_time_dim_2d(x_per, self.lorder - 1, self.lorder - 1)
        out = x_per + conv2d_nchw(self.conv1, y)
        out = mx.transpose(out, (0, 3, 2, 1))
        return x + mx.squeeze(out)


class UniDeepFsmnDilated(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, lorder: int, hidden_size: int, depth: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv = DilatedDenseNet(depth=depth, lorder=lorder, in_channels=output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        p1 = self.project(nn.relu(self.linear(x)))
        x_per = mx.transpose(mx.expand_dims(p1, axis=1), (0, 3, 2, 1))
        out = self.conv(x_per)
        out = mx.transpose(out, (0, 3, 2, 1))
        return x + mx.squeeze(out, axis=1)


class GatedFSMNDilated(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, lorder: int, hidden_size: int):
        super().__init__()
        self.to_u = FFConvM(in_channels, hidden_size, norm_klass=nn.LayerNorm, dropout=0.1)
        self.to_v = FFConvM(in_channels, hidden_size, norm_klass=nn.LayerNorm, dropout=0.1)
        self.fsmn = UniDeepFsmnDilated(in_channels, out_channels, lorder, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x_u = self.fsmn(self.to_u(x))
        x_v = self.to_v(x)
        return x_v * x_u + x


class GatedFSMNBlockDilated(nn.Module):
    def __init__(self, dim: int, inner_channels: int = 256):
        super().__init__()
        self.conv1 = [nn.Conv1d(dim, inner_channels, kernel_size=1), PReLU()]
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_fsmn = GatedFSMNDilated(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        conv1 = self.conv1[1](conv1d_ncl(self.conv1[0], nlc_to_ncl(x)))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(ncl_to_nlc(norm1))
        norm2 = self.norm2(nlc_to_ncl(seq_out))
        conv2 = conv1d_ncl(self.conv2, norm2)
        return ncl_to_nlc(conv2) + x


class FLASHShareAFFConvM(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        group_size: int = 256,
        query_key_dim: int = 128,
        expansion_factor: float = 1.0,
        causal: bool = False,
        dropout: float = 0.1,
        rotary_pos_emb: RotaryEmbedding | None = None,
        norm_klass=nn.LayerNorm,
        shift_tokens: bool = True,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens
        self.rotary_pos_emb = rotary_pos_emb
        self.dropout = Dropout(dropout)
        self.to_hidden = FFConvM(dim, hidden_dim, norm_klass=norm_klass, dropout=dropout)
        self.to_qk = FFConvM(dim, query_key_dim, norm_klass=norm_klass, dropout=dropout)
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)
        self.to_out = FFConvM(dim * 2, dim, norm_klass=norm_klass, dropout=dropout)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        residual = x
        if self.shift_tokens:
            x_shift, x_pass = mx.split(x, 2, axis=-1)
            x_shift = pad_time_dim_2d(x_shift, 1, 0)[:, :-1, :]
            x = mx.concatenate([x_shift, x_pass], axis=-1)
        v, u = mx.split(self.to_hidden(x), 2, axis=-1)
        qk = self.to_qk(x)
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, mask)
        out = (att_u * v) * mx.sigmoid(att_v * u)
        return residual + self.to_out(out)

    def cal_attention(
        self,
        x: mx.array,
        quad_q: mx.array,
        lin_q: mx.array,
        quad_k: mx.array,
        lin_k: mx.array,
        v: mx.array,
        u: mx.array,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        b, n, g = x.shape[0], x.shape[-2], self.group_size
        if mask is not None:
            lin_k = mx.where(mask[..., None], lin_k, 0.0)
        if self.rotary_pos_emb is not None:
            quad_q = self.rotary_pos_emb.rotate_queries_or_keys(quad_q)
            lin_q = self.rotary_pos_emb.rotate_queries_or_keys(lin_q)
            quad_k = self.rotary_pos_emb.rotate_queries_or_keys(quad_k)
            lin_k = self.rotary_pos_emb.rotate_queries_or_keys(lin_k)

        padding = padding_to_multiple_of(n, g)
        if padding > 0:
            quad_q = pad_time_dim_2d(quad_q, 0, padding)
            quad_k = pad_time_dim_2d(quad_k, 0, padding)
            lin_q = pad_time_dim_2d(lin_q, 0, padding)
            lin_k = pad_time_dim_2d(lin_k, 0, padding)
            v = pad_time_dim_2d(v, 0, padding)
            u = pad_time_dim_2d(u, 0, padding)
            if mask is None:
                mask = mx.ones((b, n), dtype=mx.bool_)
            mask = pad_last_dim(mask, 0, padding).astype(mx.bool_)

        groups = quad_q.shape[1] // g
        quad_q = mx.reshape(quad_q, (b, groups, g, quad_q.shape[-1]))
        quad_k = mx.reshape(quad_k, (b, groups, g, quad_k.shape[-1]))
        lin_q = mx.reshape(lin_q, (b, groups, g, lin_q.shape[-1]))
        lin_k = mx.reshape(lin_k, (b, groups, g, lin_k.shape[-1]))
        v = mx.reshape(v, (b, groups, g, v.shape[-1]))
        u = mx.reshape(u, (b, groups, g, u.shape[-1]))
        if mask is not None:
            mask = mx.reshape(mask, (b, groups, 1, g))

        sim = mx.einsum("...id,...jd->...ij", quad_q, quad_k) / g
        attn = mx.square(nn.relu(sim))
        attn = self.dropout(attn)
        if mask is not None:
            attn = mx.where(mask, attn, 0.0)

        quad_out_v = mx.einsum("...ij,...jd->...id", attn, v)
        quad_out_u = mx.einsum("...ij,...jd->...id", attn, u)
        lin_kv = mx.einsum("bgnd,bgne->bde", lin_k, v) / n
        lin_out_v = mx.einsum("bgnd,bde->bgne", lin_q, lin_kv)
        lin_ku = mx.einsum("bgnd,bgne->bde", lin_k, u) / n
        lin_out_u = mx.einsum("bgnd,bde->bgne", lin_q, lin_ku)

        out_v = mx.reshape(quad_out_v + lin_out_v, (b, groups * g, -1))[:, :n, :]
        out_u = mx.reshape(quad_out_u + lin_out_u, (b, groups * g, -1))[:, :n, :]
        return out_v, out_u


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
        self.fsmn = [GatedFSMNBlockDilated(dim) for _ in range(depth)]
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


class Encoder(nn.Module):
    def __init__(self, kernel_size: int = 2, out_channels: int = 64, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.in_channels == 1:
            x = mx.expand_dims(x, axis=1)
        return nn.relu(conv1d_ncl(self.conv1d, x))


class Decoder(nn.ConvTranspose1d):
    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 2:
            x = mx.expand_dims(x, axis=1)
        x = nlc_to_ncl(nn.ConvTranspose1d.__call__(self, ncl_to_nlc(x)))
        return mx.squeeze(x, axis=1) if x.shape[1] == 1 else mx.squeeze(x)


class MossFormerMaskNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
        self.mdl = ComputationBlock(num_blocks, out_channels, norm, skip_around_intra)
        self.conv1d_out = nn.Conv1d(out_channels, out_channels * num_spks, kernel_size=1)
        self.conv1_decoder = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = PReLU()
        self.activation = None
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
        return mx.transpose(x, (1, 0, 2, 3))


class MossFormer(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 512,
        num_blocks: int = 24,
        kernel_size: int = 16,
        norm: str = "ln",
        num_spks: int = 2,
        skip_around_intra: bool = True,
        use_global_pos_enc: bool = True,
        max_length: int = 20000,
    ):
        super().__init__()
        self.num_spks = num_spks
        self.enc = Encoder(kernel_size=kernel_size, out_channels=in_channels, in_channels=1)
        self.mask_net = MossFormerMaskNet(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            norm=norm,
            num_spks=num_spks,
            skip_around_intra=skip_around_intra,
            use_global_pos_enc=use_global_pos_enc,
            max_length=max_length,
        )
        self.dec = Decoder(out_channels, 1, kernel_size, stride=kernel_size // 2, bias=False)

    def __call__(self, input: mx.array) -> list[mx.array]:
        x = self.enc(input)
        mask = self.mask_net(x)
        sep_x = mx.stack([x] * self.num_spks) * mask
        est_source = mx.concatenate(
            [mx.expand_dims(self.dec(sep_x[i]), axis=-1) for i in range(self.num_spks)],
            axis=-1,
        )
        origin = input.shape[1]
        estimated = est_source.shape[1]
        if origin > estimated:
            est_source = mx.pad(est_source, [(0, 0), (0, origin - estimated), (0, 0)])
        else:
            est_source = est_source[:, :origin, :]
        return [est_source[:, :, spk] for spk in range(self.num_spks)]


class MossFormer2SS16K(nn.Module):
    backend = "mlx"

    def __init__(self, args):
        super().__init__()
        self.model = MossFormer(
            in_channels=args.encoder_embedding_dim,
            out_channels=args.mossformer_sequence_dim,
            num_blocks=args.num_mossformer_layer,
            kernel_size=args.encoder_kernel_size,
            num_spks=args.num_spks,
        )

    def __call__(self, inputs: mx.array) -> list[mx.array]:
        return self.model(inputs)
