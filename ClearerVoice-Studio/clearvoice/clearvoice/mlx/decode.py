"""MLX inference helpers matching the ClearerVoice decode contract."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

MAX_WAV_VALUE = 32768.0


def decode_one_audio_mossformer2_ss_16k(model, inputs, args):
    out = []
    decode_do_segment = False
    window = int(args.sampling_rate * args.decode_window)
    stride = int(window * 0.75)
    _, input_len = inputs.shape
    rms_input = (inputs ** 2).mean() ** 0.5

    if input_len > args.sampling_rate * args.one_time_decode_length:
        decode_do_segment = True

    if input_len < window:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - input_len))], axis=1)
    elif input_len < window + stride:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window + stride - input_len))], axis=1)
    elif (input_len - window) % stride != 0:
        padding = input_len - (input_len - window) // stride * stride
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], axis=1)

    inputs_mx = mx.array(np.float32(inputs))
    _, total_len = inputs_mx.shape

    if decode_do_segment:
        outputs = np.zeros((args.num_spks, total_len), dtype=np.float32)
        give_up_length = (window - stride) // 2
        current_idx = 0
        while current_idx + window <= total_len:
            tmp_input = inputs_mx[:, current_idx:current_idx + window]
            tmp_out_list = model(tmp_input)
            mx.eval(*tmp_out_list)
            for spk in range(args.num_spks):
                tmp_out = np.array(tmp_out_list[spk][0, :])
                if current_idx == 0:
                    outputs[spk, current_idx:current_idx + window - give_up_length] = tmp_out[:-give_up_length]
                else:
                    outputs[spk, current_idx + give_up_length:current_idx + window - give_up_length] = tmp_out[give_up_length:-give_up_length]
            current_idx += stride
        for spk in range(args.num_spks):
            out.append(outputs[spk, :])
    else:
        out_list = model(inputs_mx)
        mx.eval(*out_list)
        for spk in range(args.num_spks):
            out.append(np.array(out_list[spk][0, :]))

    for spk in range(args.num_spks):
        rms_out = (out[spk] ** 2).mean() ** 0.5
        if rms_out > 0:
            out[spk] = out[spk] / rms_out * rms_input
    return out


def _mossformer2_se_48k_mask(model, fbanks):
    fbanks_mx = mx.array(np.asarray(fbanks.detach().cpu(), dtype=np.float32))
    out_list = model(fbanks_mx)
    mx.eval(*out_list)
    return np.array(out_list[-1])


def decode_one_audio_mossformer2_se_48k(model, inputs, args):
    import torch
    import torchaudio
    from ..utils.misc import compute_fbank, istft, stft

    inputs = inputs[0, :]
    input_len = inputs.shape[0]
    inputs = inputs * MAX_WAV_VALUE

    def enhance_segment(audio_segment):
        fbanks = compute_fbank(audio_segment.unsqueeze(0), args)
        fbank_tr = torch.transpose(fbanks, 0, 1)
        fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
        fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
        fbank_delta = torch.transpose(fbank_delta, 0, 1)
        fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)
        fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1).unsqueeze(0)

        pred_mask = torch.from_numpy(_mossformer2_se_48k_mask(model, fbanks))
        spectrum = stft(audio_segment, args)
        pred_mask = pred_mask.permute(2, 1, 0)
        masked_spec = spectrum * pred_mask
        masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]
        return istft(masked_spec_complex, args, len(audio_segment))

    if input_len > args.sampling_rate * args.one_time_decode_length:
        window = int(args.sampling_rate * args.decode_window)
        stride = int(window * 0.75)
        total_len = inputs.shape[0]

        if total_len < window:
            inputs = np.concatenate([inputs, np.zeros(window - total_len)], 0)
        elif total_len < window + stride:
            inputs = np.concatenate([inputs, np.zeros(window + stride - total_len)], 0)
        elif (total_len - window) % stride != 0:
            padding = total_len - (total_len - window) // stride * stride
            inputs = np.concatenate([inputs, np.zeros(padding)], 0)

        audio = torch.from_numpy(np.float32(inputs))
        total_len = audio.shape[0]
        outputs = torch.from_numpy(np.zeros(total_len, dtype=np.float32))
        give_up_length = (window - stride) // 2
        current_idx = 0
        while current_idx + window <= total_len:
            audio_segment = audio[current_idx:current_idx + window]
            output_segment = enhance_segment(audio_segment)
            if current_idx == 0:
                outputs[current_idx:current_idx + window - give_up_length] = output_segment[:-give_up_length]
            else:
                outputs[current_idx + give_up_length:current_idx + window - give_up_length] = output_segment[give_up_length:-give_up_length]
            current_idx += stride
    else:
        audio = torch.from_numpy(np.float32(inputs))
        outputs = enhance_segment(audio)

    return outputs.numpy() / MAX_WAV_VALUE
