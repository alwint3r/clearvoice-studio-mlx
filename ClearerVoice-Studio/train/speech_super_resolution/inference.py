import torch
import sys
import os
import argparse
import numpy as np
from pydub import AudioSegment
sys.path.append(os.path.dirname(sys.path[0]))

from utils.misc import reload_for_eval, load_config_json, combine_config_and_args
from utils.decode import decode_one_audio
from dataloader.dataloader import DataReader
import yamlargparse
import soundfile as sf
import warnings
from networks import network_wrapper

warnings.filterwarnings("ignore")

def inference(args):
    device = torch.device('cuda') if args.use_cuda==1 else torch.device('cpu')
    print(device)
    print('creating model...')
    models = network_wrapper(args).models
    for model in models:
        model.to(device)

    print('loading model ...')
    reload_for_eval(models[0], args.checkpoint_dir, args.use_cuda, model_key='mossformer')
    reload_for_eval(models[1], args.checkpoint_dir, args.use_cuda, model_key='generator')
    models[0].eval()
    models[1].eval()
    with torch.no_grad():

        data_reader = DataReader(args)
        output_wave_dir = args.output_dir
        if not os.path.isdir(output_wave_dir):
            os.makedirs(output_wave_dir)
        num_samples = len(data_reader)
        print('Decoding...')
        for idx in range(num_samples):
            input_audio, wav_id, input_len, scalar, data_info = data_reader[idx]
            if input_audio is None: continue
            # Read the audio, waveform ID, and audio length from the data reader
            print(f'audio: {wav_id}')
            output_audio = decode_one_audio(models, device, input_audio[0], args)             
            output_audio = output_audio[:input_len] * scalar
            if data_info['sample_width'] == 4: ##32 bit float
                MAX_WAV_VALUE = 2147483648.0
                np_type = np.int32
            elif data_info['sample_width'] == 2: ##16 bit int
                MAX_WAV_VALUE = 32768.0
                np_type = np.int16
            else:
                data_info['sample_width'] = 2 ##16 bit int
                MAX_WAV_VALUE = 32768.0
                np_type = np.int16
                        
            output_audio = output_audio * MAX_WAV_VALUE
            output_audio = output_audio.astype(np_type)
            audio_segment = AudioSegment(
                output_audio.tobytes(),  # Raw audio data as bytes
                frame_rate=data_info['sample_rate'],  # Sample rate
                sample_width=data_info['sample_width'],          # No. bytes per sample
                channels=data_info['channels']               # No. channels
            )
            audio_format = 'ipod' if data_info['ext'] in ['m4a', 'aac'] else data_info['ext']
            output_path = os.path.join(output_wave_dir, wav_id)
            audio_segment.export(output_path, format=audio_format)

            #sf.write(os.path.join(output_wave_dir, wav_id), output_audio, args.sampling_rate)
    print('Done!')
if __name__ == "__main__":
    # parser = argparse.ArgumentParser('PyTorch Version ENhancement')
    """
    Loads the arguments for the speech super-resolution task using a YAML config file.
    Sets the configuration path and parses all the required parameters such as
    input/output paths, model settings, and FFT parameters.
    """
    #config_path = 'config/inference/' + model_name + '.yaml'
    parser = yamlargparse.ArgumentParser("Settings")

    # General model and inference settings
    parser.add_argument('--config', help='Config file path', action=yamlargparse.ActionConfigFile)
    parser.add_argument('--config_json', type=str, help='Path to the config.json file')
    parser.add_argument('--mode', type=str, default='inference', help='Modes: train or inference')
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='checkpoints/MossFormer2_SR_48K', help='Checkpoint directory')
    parser.add_argument('--input-path', dest='input_path', type=str, help='Path for low-resolution audio input')
    parser.add_argument('--output-dir', dest='output_dir', type=str, help='Directory for enhanced high-resolution audio output')
    parser.add_argument('--use-cuda', dest='use_cuda', default=1, type=int, help='Enable CUDA (1=True, 0=False)')
    parser.add_argument('--num-gpu', dest='num_gpu', type=int, default=1, help='Number of GPUs to use')

    # Model-specific settings
    parser.add_argument('--network', type=str, help='Select SR model(currently supports MossFormer2_SR_48K)')
    parser.add_argument('--sampling-rate', dest='sampling_rate', type=int, default=48000, help='Sampling rate')
    parser.add_argument('--one-time-decode-length', dest='one_time_decode_length', type=float, default=60.0, help='Max segment length for one-pass decoding')
    parser.add_argument('--decode-window', dest='decode_window', type=float, default=1.0, help='Decoding chunk size')

    # Parse arguments from the config file
    #args = parser.parse_args(['--config', self.config_path])
    args = parser.parse_args()
    json_config = load_config_json(args.config_json)
    args = combine_config_and_args(json_config, args)
    args = argparse.Namespace(**args)    
    print(args)

    inference(args)
