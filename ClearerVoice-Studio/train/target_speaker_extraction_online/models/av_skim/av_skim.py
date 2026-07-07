from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from torchvision import transforms
import numpy as np

EPS = 1e-8


class av_skim(nn.Module):
    def __init__(self, args):
        super(av_skim, self).__init__()
        N, L = args.network_audio.input_dim, args.network_audio.window
        self.args = args

        self.encoder = Encoder(L, N)
        self.separator = SkiMSeparator(args)
        self.decoder = Decoder(N, L)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture, visual, reference=None):

        if self.args.evaluate_only and self.args.network_audio.tcn_attractor:
            stride = int(self.args.network_audio.window/2)
            states=None
            v_emb=self.separator.forward_streaming_v_emb(visual)
            est_source = torch.zeros_like(mixture)            

            for itr in range(int(np.floor(mixture.shape[1]/stride))-1):
                if itr >=2:
                    reference = est_source[:,:itr*stride]
                    # reference = est_source[:,(itr-2)*stride:itr*stride]
                mixture_tmp = mixture[:,itr*stride:(itr+2)*stride].clone()
                mixture_w_tmp = self.encoder(mixture_tmp)
                visual_tmp = v_emb[:,:,itr].clone().unsqueeze(2)
                est_mask, states = self.separator.forward_streaming_av(mixture_w_tmp, visual_tmp, states, reference)
                est_source_tmp = self.decoder(mixture_w_tmp, est_mask)
                est_source[:,itr*stride:(itr+2)*stride] +=est_source_tmp

            return est_source

        else:
            mixture_w = self.encoder(mixture)
            est_mask = self.separator(mixture_w, visual, reference)

            est_source = self.decoder(mixture_w, est_mask)
            # T changed after conv1d in encoder, fix it here
            T_origin = mixture.size(-1)
            T_conv = est_source.size(-1)
            est_source = F.pad(est_source, (0, T_origin - T_conv))
            return est_source


class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        est_source = mixture_w * est_mask  # [M,  N, K]
        est_source = torch.transpose(est_source, 2, 1) # [M,  K, N]
        est_source = self.basis_signals(est_source)  # [M,  K, L]
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T
        return est_source



class SkiMSeparator(nn.Module):
    def __init__(
        self,
        args,
        nonlinear: str = "relu",
        dropout: float = 0.0,
        mem_type: str = "hc",
        seg_overlap: bool = False,
    ):
        super().__init__()

        self.args=args
        causal = args.causal

        if mem_type not in ("hc", "h", "c", "id", None):
            raise ValueError("Not supporting mem_type={}".format(mem_type))

        self.skim = SkiM(
            args=args,
            input_size=args.network_audio.input_dim,
            hidden_size=args.network_audio.unit,
            output_size=args.network_audio.input_dim,
            dropout=dropout,
            num_blocks=args.network_audio.layer,
            bidirectional=(not causal),
            norm_type="cLN" if causal else "gLN",
            segment_size=args.network_audio.segment_size,
            seg_overlap=seg_overlap,
            mem_type=mem_type,
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

        self.v_ds = nn.Conv1d(args.network_audio.img_emb_size, args.network_audio.input_dim, 1)
        self.av_conv = nn.Conv1d(args.network_audio.input_dim*2, args.network_audio.input_dim, 1)

        if self.args.network_audio.tcn_attractor:
            from .speaker_encoder_tcn import SpeakerEncoder
            self.spk_encoder = nn.Sequential(Encoder(args.network_audio.window, args.network_audio.input_dim), SpeakerEncoder(args, B=args.network_audio.input_dim, causal=self.args.causal))
            self.av_conv_speech = nn.Conv1d(args.network_audio.input_dim*2, args.network_audio.input_dim, 1, bias=False)  


    def forward(self, audio, visual, reference):
        visual = self.v_ds(visual)
        v_length = visual.shape[2]
        a_length = audio.shape[2]
        visual = torch.repeat_interleave(visual.unsqueeze(3), repeats=80, dim=3)    # 80: 25fps  # 160: 12.5fps   # 400: 5fps
        visual = visual.reshape(visual.shape[0],visual.shape[1],-1)
        visual = F.pad(visual,(0,a_length-visual.shape[2]),"constant",0)
        audio = torch.cat((audio, visual),1)
        audio  = self.av_conv(audio)

        if self.args.network_audio.tcn_attractor:
            if reference == None:
                reference = torch.zeros_like(audio).cuda()
            else:
                reference = self.spk_encoder(reference)
            audio = torch.cat((audio, reference),1)
            audio  = self.av_conv_speech(audio)

        processed = self.skim(audio.transpose(1,2))

        mask = self.nonlinear(processed).transpose(1,2)
        return mask

    def forward_streaming(self, input_frame, states=None):
        processed, states = self.skim.forward_stream(audio.transpose(1,2), states=states)
        mask = self.nonlinear(processed).transpose(1,2)
        return mask, states

    def forward_streaming_av(self, audio, visual, states=None, reference=None):
        audio = torch.cat((audio, visual),1)
        audio  = self.av_conv(audio)

        if self.args.network_audio.tcn_attractor:
            if reference == None:
                reference = torch.zeros_like(audio).cuda()
                self.spk_state = None
            else:
                reference = self.spk_encoder[0](reference)
                # reference, self.spk_state = self.spk_encoder[1](reference, state = self.spk_state)
                reference, _ = self.spk_encoder[1](reference)
                reference = reference[:,:,-1].unsqueeze(2)
            audio = torch.cat((audio, reference),1)
            audio  = self.av_conv_speech(audio)

        processed, states = self.skim.forward_stream(audio.transpose(1,2), states=states)
        mask = self.nonlinear(processed).transpose(1,2)

        return mask, states

    def forward_streaming_v_emb(self, visual):
        visual = self.v_ds(visual)
        visual = torch.repeat_interleave(visual.unsqueeze(3), repeats=80, dim=3)    # 80: 25fps  # 160: 12.5fps   # 400: 5fps
        visual = visual.reshape(visual.shape[0],visual.shape[1],-1)
        visual = F.pad(visual,(0,200),"constant",0)
        return visual


class MemLSTM(nn.Module):
    """the Mem-LSTM of SkiM

    args:
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the LSTM layers are bidirectional.
            Default is False.
        mem_type: 'hc', 'h', 'c' or 'id'.
            It controls whether the hidden (or cell) state of
            SegLSTM will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states will
            be identically returned.
        norm_type: gLN, cLN. cLN is for causal implementation.
    """

    def __init__(
        self,
        hidden_size,
        dropout=0.0,
        bidirectional=False,
        mem_type="hc",
        norm_type="cLN",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_size = (int(bidirectional) + 1) * hidden_size
        self.mem_type = mem_type

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
        ], f"only support 'hc', 'h', 'c' and 'id', current type: {mem_type}"

        if mem_type in ["hc", "h"]:
            self.h_net = SingleRNN(
                "LSTM",
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.h_norm = choose_norm(
                norm_type=norm_type, channel_size=self.input_size, shape="BTD"
            )
        if mem_type in ["hc", "c"]:
            self.c_net = SingleRNN(
                "LSTM",
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.c_norm = choose_norm(
                norm_type=norm_type, channel_size=self.input_size, shape="BTD"
            )

    def extra_repr(self) -> str:
        return f"Mem_type: {self.mem_type}, bidirectional: {self.bidirectional}"

    def forward(self, hc, S):
        # hc = (h, c), tuple of hidden and cell states from SegLSTM
        # shape of h and c: (d, B*S, H)
        # S: number of segments in SegLSTM

        if self.mem_type == "id":
            ret_val = hc
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
        else:
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
            h = h.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH
            c = c.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH
            if self.mem_type == "hc":
                h = h + self.h_norm(self.h_net(h)[0])
                c = c + self.c_norm(self.c_net(c)[0])
            elif self.mem_type == "h":
                h = h + self.h_norm(self.h_net(h)[0])
                c = torch.zeros_like(c)
            elif self.mem_type == "c":
                h = torch.zeros_like(h)
                c = c + self.c_norm(self.c_net(c)[0])

            h = h.view(B * S, d, H).transpose(1, 0).contiguous()
            c = c.view(B * S, d, H).transpose(1, 0).contiguous()
            ret_val = (h, c)

        if not self.bidirectional:
            # for causal setup
            causal_ret_val = []
            for x in ret_val:
                x = x.transpose(1, 0).contiguous().view(B, S, d * H)
                x_ = torch.zeros_like(x)
                x_[:, 1:, :] = x[:, :-1, :]
                x_ = x_.view(B * S, d, H).transpose(1, 0).contiguous()
                causal_ret_val.append(x_)
            ret_val = tuple(causal_ret_val)

        return ret_val

    def forward_one_step(self, hc, state):
        if self.mem_type == "id":
            pass
        else:
            h, c = hc
            d, B, H = h.shape
            h = h.transpose(1, 0).contiguous().view(B, 1, d * H)  # B, 1, dH
            c = c.transpose(1, 0).contiguous().view(B, 1, d * H)  # B, 1, dH
            if self.mem_type == "hc":
                h_tmp, state[0] = self.h_net(h, state[0])
                h = h + self.h_norm(h_tmp)
                c_tmp, state[1] = self.c_net(c, state[1])
                c = c + self.c_norm(c_tmp)
            elif self.mem_type == "h":
                h_tmp, state[0] = self.h_net(h, state[0])
                h = h + self.h_norm(h_tmp)
                c = torch.zeros_like(c)
            elif self.mem_type == "c":
                h = torch.zeros_like(h)
                c_tmp, state[1] = self.c_net(c, state[1])
                c = c + self.c_norm(c_tmp)
            h = h.transpose(1, 0).contiguous()
            c = c.transpose(1, 0).contiguous()
            hc = (h, c)

        return hc, state


class SegLSTM(nn.Module):
    """the Seg-LSTM of SkiM

    args:
        input_size: int, dimension of the input feature.
            The input should have shape (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the LSTM layers are bidirectional.
            Default is False.
        norm_type: gLN, cLN. cLN is for causal implementation.
    """

    def __init__(
        self, input_size, hidden_size, dropout=0.0, bidirectional=False, norm_type="cLN"
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)
        self.norm = choose_norm(
            norm_type=norm_type, channel_size=input_size, shape="BTD"
        )

    def forward(self, input, hc):
        # input shape: B, T, H

        B, T, H = input.shape

        if hc is None:
            # In fist input SkiM block, h and c are not available
            d = self.num_direction
            h = torch.zeros(d, B, self.hidden_size, dtype=input.dtype).to(input.device)
            c = torch.zeros(d, B, self.hidden_size, dtype=input.dtype).to(input.device)
        else:
            h, c = hc

        output, (h, c) = self.lstm(input, (h, c))
        output = self.dropout(output)
        output = self.proj(output.contiguous().view(-1, output.shape[2])).view(
            input.shape
        )
        output = input + self.norm(output)

        return output, (h, c)


class SkiM(nn.Module):
    """Skipping Memory Net

    args:
        input_size: int, dimension of the input feature.
            Input shape shoud be (batch, length, input_size)
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_blocks: number of basic SkiM blocks
        segment_size: segmentation size for splitting long features
        bidirectional: bool, whether the RNN layers are bidirectional.
        mem_type: 'hc', 'h', 'c', 'id' or None.
            It controls whether the hidden (or cell) state of SegLSTM
            will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states will
            be identically returned.
            When mem_type is None, the MemLSTM will be removed.
        norm_type: gLN, cLN. cLN is for causal implementation.
        seg_overlap: Bool, whether the segmentation will reserve 50%
            overlap for adjacent segments.Default is False.
    """

    def __init__(
        self,
        args,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0,
        num_blocks=2,
        segment_size=20,
        bidirectional=True,
        mem_type="hc",
        norm_type="gLN",
        seg_overlap=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.segment_size = segment_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.mem_type = mem_type
        self.norm_type = norm_type
        self.seg_overlap = seg_overlap

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
            None,
        ], f"only support 'hc', 'h', 'c', 'id', and None, current type: {mem_type}"

        self.seg_lstms = nn.ModuleList([])
        for i in range(num_blocks):
            self.seg_lstms.append(
                SegLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    norm_type=norm_type,
                )
            )
        if self.mem_type is not None:
            self.mem_lstms = nn.ModuleList([])
            for i in range(num_blocks - 1):
                self.mem_lstms.append(
                    MemLSTM(
                        hidden_size,
                        dropout=dropout,
                        bidirectional=bidirectional,
                        mem_type=mem_type,
                        norm_type=norm_type,
                    )
                )
        self.output_fc = nn.Sequential(
            nn.PReLU(), nn.Conv1d(input_size, output_size, 1)
        )
        
        self.args=args

    def forward(self, input):
        # input shape: B, T (S*K), D
        B, T, D = input.shape

        if self.seg_overlap:
            input, rest = split_feature(
                input.transpose(1, 2), segment_size=self.segment_size
            )  # B, D, K, S
            input = input.permute(0, 3, 2, 1).contiguous()  # B, S, K, D
        else:
            input, rest = self._padfeature(input=input)
            input = input.view(B, -1, self.segment_size, D)  # B, S, K, D
        B, S, K, D = input.shape

        assert K == self.segment_size

        output = input.view(B * S, K, D).contiguous()  # BS, K, D
        hc = None
        for i in range(self.num_blocks):
            output, hc = self.seg_lstms[i](output, hc)  # BS, K, D
            if self.mem_type and i < self.num_blocks - 1:
                hc = self.mem_lstms[i](hc, S)
                pass

        if self.seg_overlap:
            output = output.view(B, S, K, D).permute(0, 3, 2, 1)  # B, D, K, S
            output = merge_feature(output, rest)  # B, D, T
            output = self.output_fc(output).transpose(1, 2)

        else:
            output = output.view(B, S * K, D)[:, :T, :]  # B, T, D

            output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)
                
        return output

    def _padfeature(self, input):
        B, T, D = input.shape
        rest = self.segment_size - T % self.segment_size

        if rest > 0:
            input = torch.nn.functional.pad(input, (0, 0, 0, rest))
        return input, rest

    def forward_stream(self, input_frame, states):
        # input_frame # B, 1, N

        B, _, N = input_frame.shape

        def empty_seg_states():
            shp = (1, B, self.hidden_size)
            return (
                torch.zeros(*shp, device=input_frame.device, dtype=input_frame.dtype),
                torch.zeros(*shp, device=input_frame.device, dtype=input_frame.dtype),
            )

        B, _, N = input_frame.shape
        if not states:
            states = {
                "current_step": 0,
                "seg_state": [empty_seg_states() for i in range(self.num_blocks)],
                "mem_state": [[None, None] for i in range(self.num_blocks - 1)],
            }

        output = input_frame

        if states["current_step"] and (states["current_step"]) % self.segment_size == 0:
            tmp_states = [empty_seg_states() for i in range(self.num_blocks)]
            for i in range(self.num_blocks - 1):
                tmp_states[i + 1], states["mem_state"][i] = self.mem_lstms[
                    i
                ].forward_one_step(states["seg_state"][i], states["mem_state"][i])

            states["seg_state"] = tmp_states

        for i in range(self.num_blocks):
            output, states["seg_state"][i] = self.seg_lstms[i](
                output, states["seg_state"][i]
            )

        states["current_step"] += 1

        output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)

        return output, states

def choose_norm(norm_type, channel_size, shape="BDT"):
    """The input of normalization will be (M, C, K), where M is batch size.

    C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size, shape=shape)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size, shape=shape)
    elif norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)
    elif norm_type == "GN":
        return nn.GroupNorm(1, channel_size, eps=1e-8)
    else:
        raise ValueError("Unsupported normalization type")


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)."""

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            cLN_y: [M, N, K]
        """

        assert y.dim() == 3

        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            cLN_y = cLN_y.transpose(1, 2).contiguous()

        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)."""

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            gLN_y: [M, N, K]
        """
        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = y.mean(dim=(1, 2), keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=(1, 2), keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            gLN_y = gLN_y.transpose(1, 2).contiguous()
        return gLN_y


class SingleRNN(nn.Module):
    """Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False
    ):
        super().__init__()

        rnn_type = rnn_type.upper()

        assert rnn_type in [
            "RNN",
            "LSTM",
            "GRU",
        ], f"Only support 'RNN', 'LSTM' and 'GRU', current type: {rnn_type}"

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input, state=None):
        # input shape: batch, seq, dim
        # input = input.to(device)
        output = input
        rnn_output, state = self.rnn(output, state)
        rnn_output = self.dropout(rnn_output)
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(output.shape)
        return rnn_output, state


def _pad_segment(input, segment_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
        input = torch.cat([input, pad], 2)

    pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest


def split_feature(input, segment_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = _pad_segment(input, segment_size)
    batch_size, dim, seq_len = input.shape
    segment_stride = segment_size // 2

    segments1 = (
        input[:, :, :-segment_stride]
        .contiguous()
        .view(batch_size, dim, -1, segment_size)
    )
    segments2 = (
        input[:, :, segment_stride:]
        .contiguous()
        .view(batch_size, dim, -1, segment_size)
    )
    segments = (
        torch.cat([segments1, segments2], 3)
        .view(batch_size, dim, -1, segment_size)
        .transpose(2, 3)
    )

    return segments.contiguous(), rest


def merge_feature(input, rest):
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, segment_size, _ = input.shape
    segment_stride = segment_size // 2
    input = (
        input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)
    )  # B, N, K, L

    input1 = (
        input[:, :, :, :segment_size]
        .contiguous()
        .view(batch_size, dim, -1)[:, :, segment_stride:]
    )
    input2 = (
        input[:, :, :, segment_size:]
        .contiguous()
        .view(batch_size, dim, -1)[:, :, :-segment_stride]
    )

    output = input1 + input2
    if rest > 0:
        output = output[:, :, :-rest]

    return output.contiguous()  # B, N, T


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long().cuda()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result