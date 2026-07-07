#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

EPS = 1e-8
import copy
def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class SpeakerEncoder(nn.Module):
    def __init__(self, args, B=256, H=256, causal=True):
        super(SpeakerEncoder, self).__init__()
        self.causal = causal
        if causal:
            self.layer_norm = cumulative_ChannelWiseLayerNorm(args,B)
        else:
            self.layer_norm = ChannelWiseLayerNorm(B)
        self.bottleneck_conv1x1 = nn.Conv1d(B, B, 1, bias=False)

        self.mynet = nn.Sequential(
            ResBlock(args, B, B, residual=True, causal=causal),
            ResBlock(args, B, H, residual=True, causal=causal),
            ResBlock(args, H, H, residual=False, causal=causal),
            # nn.Dropout(0.2),
            nn.Conv1d(H, B, 1, bias=False)
        )

        self.args = args
        if causal:
            self.lstm = torch.nn.LSTM(input_size=B,hidden_size=B,num_layers=1,batch_first=True,bidirectional=False)
            self.lstm_state = None


    def forward(self, x, state=None):
        if self.args.evaluate_only:
            B, D, T = x.size()
            x = self.layer_norm(x)
            x = self.bottleneck_conv1x1(x)
            x = self.mynet(x)
            output, state = self.lstm(x.transpose(1,2), state)
            x = output.transpose(1,2)
            return x, state
        else:
            B, D, T = x.size()

            x = self.layer_norm(x)
            x = self.bottleneck_conv1x1(x)
            x = self.mynet(x)

            # X: B, D, T
            if self.causal:
                output, (hn, cn) = self.lstm(x.transpose(1,2))
                x = output.transpose(1,2)
            else:
                x = F.adaptive_avg_pool1d(x,1)
            return x


class ResBlock(nn.Module):
    def __init__(self, args, in_dims, out_dims, residual, causal):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        if causal:
            self.norm1 = cumulative_ChannelWiseLayerNorm(args, out_dims)
            self.norm2 = cumulative_ChannelWiseLayerNorm(args, out_dims)
        else:
            self.norm1 = nn.GroupNorm(1, out_dims, eps=1e-8)
            self.norm2 = nn.GroupNorm(1, out_dims, eps=1e-8)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        # self.mp = nn.MaxPool1d(3)
        self.residual = residual
        if self.residual:
            if in_dims != out_dims:
                self.downsample = True
                self.conv_downsample = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
            else:
                self.downsample = False

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.residual:
            if self.downsample:
                residual = self.conv_downsample(residual)
            x = x + residual
        x = self.prelu2(x)
        # x = self.mp(x)
        return x




class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


class cumulative_ChannelWiseLayerNorm(nn.Module):
    def __init__(self, args, dimension, eps = 1e-8, trainable=True):
        super(cumulative_ChannelWiseLayerNorm, self).__init__()
        
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

        self.args = args


    def forward(self, input):
        return self.cal_norm(input)

    def cal_norm(self, input):
        channel = input.size(1)
        time_step = input.size(2)
        
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        output = x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())

        return output


# class cumulative_ChannelWiseLayerNorm(nn.Module):
#     def __init__(self, args, dimension, eps = 1e-8, trainable=True):
#         super(cumulative_ChannelWiseLayerNorm, self).__init__()
        
#         self.eps = eps
#         if trainable:
#             self.gain = nn.Parameter(torch.ones(1, dimension, 1))
#             self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
#         else:
#             self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
#             self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

#         self.args = args

#         if self.args.evaluate:
#             self.mean_num = 0
#             self.pre_mean = 0
#             self.pre_var = 0

#     def forward(self, input):
#         # input size: (Batch, Freq, Time)
#         # cumulative mean for each time step
        
#         if self.args.evaluate:
#             channel = input.size(1)
#             time_step = input.size(2)
#             if self.mean_num ==0:
#                 mean = input.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
#                 var = (torch.pow(input-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
#             else:
#                 mean = self.mean_num * self.pre_mean + input.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
#                 mean /= (self.mean_num + time_step*channel)
                
#                 var = self.mean_num * self.pre_var + torch.pow(input-mean, 2).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
#                 var /= (self.mean_num + time_step*channel)

#             self.mean_num += time_step*channel
#             self.pre_mean = mean
#             self.pre_var = var
#             output = self.gain * (input - mean) / torch.pow(var + self.eps, 0.5) + self.bias
#         else:
#             channel = input.size(1)
#             time_step = input.size(2)
            
#             step_sum = input.sum(1)  # B, T
#             step_pow_sum = input.pow(2).sum(1)  # B, T
#             cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
#             cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
            
#             entry_cnt = np.arange(channel, channel*(time_step+1), channel)
#             entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
#             entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
            
#             cum_mean = cum_sum / entry_cnt  # B, T
#             cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
#             cum_std = (cum_var + self.eps).sqrt()  # B, T
            
#             cum_mean = cum_mean.unsqueeze(1)
#             cum_std = cum_std.unsqueeze(1)
            
#             x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
#             output = x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())

#         return output