from librosa.util import normalize
from scipy.io.wavfile import read
import scipy
import wave
from pydub import AudioSegment

import numpy as np
import math, os, csv
import torchaudio
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import soundfile as sf
from torch.utils.data import Dataset
import torch.utils.data as data
import os 
import sys
sys.path.append(os.path.dirname(__file__))
from pydub import AudioSegment
from dataloader.misc import read_and_config_file, get_file_extension
import librosa
import random
EPS = 1e-6
MAX_WAV_VALUE_16B = 32768.0
MAX_WAV_VALUE_32B = 2147483648.0
   
def read_audio(file_path):
    """
    Use AudioSegment to load audio from all supported audio input format
    """
    
    try:
        audio = AudioSegment.from_file(file_path)
        return audio
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
        
def audioread(path, sampling_rate, use_norm):
    """
    Reads an audio file from the specified path, normalizes the audio,
    resamples it to the desired sampling rate (if necessary), and ensures it is single-channel.

    Parameters:
    path (str): The file path of the audio file to be read.
    sampling_rate (int): The target sampling rate for the audio.
    use_norm (bool): The flag for specifying whether using input audio normalization

    Returns:
    numpy.ndarray: The processed audio data, normalized, resampled (if necessary),
                   and converted to mono (if the input audio has multiple channels).
    """
    
    # Read audio data and its sample rate from the file.
    audio_info = {}
    ext = get_file_extension(path).replace('.', '')
    audio_info['ext']=ext
    data = read_audio(path)
    if data is None: return None, None, None
   
    audio_info['sample_rate'] = data.frame_rate
    audio_info['channels'] = data.channels
    audio_info['sample_width'] = data.sample_width

    data_array = np.array(data.get_array_of_samples())
    if max(data_array) > MAX_WAV_VALUE_16B:
        audio_np = data_array / MAX_WAV_VALUE_32B
    else:
        audio_np = data_array / MAX_WAV_VALUE_16B

    audios = []
    # Check if the audio is stereo
    if audio_info['channels'] == 2:
        audios.append(audio_np[::2])  # Even indices (left channel)
        audios.append(audio_np[1::2])  # Odd indices (right channel)
    else:
        audios.append(audio_np)
    
    # Normalize the audio data.
    audios_normed = []
    scalars = []
    for audio in audios:
        if use_norm:
            audio_normed, scalar = audio_norm(audio)
            audios_normed.append(audio_normed)
            scalars.append(scalar)
        else:
            audios_normed.append(audio)
            scalars.append(1)
    # Resample the audio if the sample rate is different from the target sampling rate.
    if audio_info['sample_rate'] != sampling_rate:
        index = 0
        for audio_normed in audios_normed:
            audios_normed[index] = librosa.resample(audio_normed, orig_sr=audio_info['sample_rate'], target_sr=sampling_rate)
            index = index + 1
    
    # Return the processed audio data.
    return audios_normed, scalars, audio_info

def audio_norm(x):
    """
    Normalizes the input audio signal to a target Root Mean Square (RMS) level, 
    applying two stages of scaling. This ensures the audio signal is neither too quiet 
    nor too loud, keeping its amplitude consistent.

    Parameters:
    x (numpy.ndarray): Input audio signal to be normalized.

    Returns:
    numpy.ndarray: Normalized audio signal.
    """
    
    # Compute the root mean square (RMS) of the input audio signal.
    rms = (x ** 2).mean() ** 0.5
    
    # Calculate the scalar to adjust the signal to the target level (-25 dB).
    scalar = 10 ** (-25 / 20) / (rms + EPS)
    
    # Scale the input audio by the computed scalar.
    x = x * scalar
    
    # Compute the power of the scaled audio signal.
    pow_x = x ** 2
    
    # Calculate the average power of the audio signal.
    avg_pow_x = pow_x.mean()
    
    # Compute RMS only for audio segments with higher-than-average power.
    rmsx = pow_x[pow_x > avg_pow_x].mean() ** 0.5
    
    # Calculate another scalar to further normalize based on higher-power segments.
    scalarx = 10 ** (-25 / 20) / (rmsx + EPS)
    
    # Apply the second scalar to the audio.
    x = x * scalarx
    
    # Return the doubly normalized audio signal.
    return x, 1/(scalar * scalarx + EPS)

class DataReader(object):
    """
    A class for reading audio data from a list of files, normalizing it, 
    and extracting features for further processing. It supports extracting 
    features from each file, reshaping the data, and returning metadata 
    like utterance ID and data length.

    Parameters:
    args: Arguments containing the input path and target sampling rate.

    Attributes:
    file_list (list): A list of audio file paths to process.
    sampling_rate (int): The target sampling rate for audio files.
    """

    def __init__(self, args):
        # Read and configure the file list from the input path provided in the arguments.
        # The file list is decoded, if necessary.
        self.file_list = read_and_config_file(args, args.input_path, decode=True)
        
        # Store the target sampling rate.
        self.sampling_rate = args.sampling_rate

        # Store the args file
        self.args = args

    def __len__(self):
        """
        Returns the number of audio files in the file list.

        Returns:
        int: Number of files to process.
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Retrieves the features of the audio file at the given index.

        Parameters:
        index (int): Index of the file in the file list.

        Returns:
        tuple: Features (inputs, utterance ID, data length) for the selected audio file.
        """
        return self.extract_feature(self.file_list[index])

    def extract_feature(self, path):
        """
        Extracts features from the given audio file path.

        Parameters:
        path (str): The file path of the audio file.

        Returns:
        inputs (numpy.ndarray): Reshaped audio data for further processing.
        utt_id (str): The unique identifier of the audio file, usually the filename.
        length (int): The length of the original audio data.
        """
        # Extract the utterance ID from the file path (usually the filename).
        utt_id = os.path.basename(path)
        use_norm = False
        
        #We suggest to use norm for 'FRCRN_SE_16K' and 'MossFormer2_SS_16K' models
        if self.args.network in ['FRCRN_SE_16K','MossFormer2_SS_16K'] :
            use_norm = True
            
        # Read and normalize the audio data, converting it to float32 for processing.
        audios_norm, scalars, audio_info = audioread(path, self.sampling_rate, use_norm)
        if audios_norm is None: return None, None, None, None, None

        if self.args.network in ['MossFormer2_SR_48K']:
            audio_info['sample_rate'] = self.sampling_rate
            
        for i in range(len(audios_norm)):
            audios_norm[i] = audios_norm[i].astype(np.float32)
            # Reshape the data to ensure it's in the format [1, data_length].
            audios_norm[i] = np.reshape(audios_norm[i], [1, audios_norm[i].shape[0]])
        
        # Return the reshaped audio data, utterance ID, and the length of the original data.
        return audios_norm, utt_id, audios_norm[0].shape[1], scalars, audio_info

def load_wav(full_path):
    try:
        sampling_rate, data = read(full_path)
        if max(data.shape) / sampling_rate < 0.5:
            return None, None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

    if len(data.shape) > 1:
        if data.shape[1] <= 2:
            data = data[...,0]
        else:
            data = data[0,...]
    return data / MAX_WAV_VALUE_16B, sampling_rate

def get_wave_duration(file_path):
    """
    Gets the duration of a WAV file in seconds.

    :param file_path: Path to the WAV file.
    :return: Duration of the WAV file in seconds.
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            # Get the number of frames
            num_frames = wf.getnframes()
            # Get the frame rate
            frame_rate = wf.getframerate()
            # Calculate duration
            duration = num_frames / float(frame_rate)
            return duration, frame_rate, num_frames
    except wave.Error as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None

def read_audio_segment(file_path, start_ms, end_ms):
    """
    Reads a segment from a WAV file and returns the raw data and its properties.

    :param file_path: Path to the WAV file.
    :param start_ms: Start time of the segment in milliseconds.
    :param end_ms: End time of the segment in milliseconds.
    :return: A tuple containing the raw audio data, frame rate, sample width, and number of channels.
    """
    #start_time = time.time()
    try:
        # Load the audio file
        audio = AudioSegment.from_wav(file_path)
        # Extract the segment
        segment = audio[start_ms:end_ms]
        # Get raw audio data
        raw_data = segment.raw_data
        # Get audio properties
        frame_rate = segment.frame_rate
        sample_width = segment.sample_width
        channels = segment.channels
        # Create NumPy array from the raw audio data
        audio_array = np.frombuffer(raw_data, dtype=np.int16)

        # If stereo, reshape the array to have a second dimension
        if channels > 1:
            audio_array = audio_array.reshape((-1, channels))
            audio_array = audio_array[...,0]
        '''
        if frame_rate !=48000:
            audio_array = audio_array/MAX_WAV_VALUE_16B
            audio_array = librosa.resample(audio_array, frame_rate, 48000)
            audio_array = audio_array * MAX_WAV_VALUE_16B
            frame_rate = 48000
        '''
        #end_time = time.time()
        #time_taken = end_time - start_time

        #print(f"Successfully read segment from {start_ms}ms to {end_ms}ms in {time_taken:.4f} seconds")
        return audio_array / MAX_WAV_VALUE_16B #, frame_rate #, sample_width, channels
    except Exception as e:
        print(f"An error occurred: {e}")
        return None#, None #, None, None

def resample(audio, sr_in, sr_out, target_len=None):
    #audio = audio / MAX_WAV_VALUE_16B
    #audio = normalize(audio) * 0.95
    if target_len is not None:
        audio = scipy.signal.resample(audio, target_len)
        return audio
    resample_factor = sr_out / sr_in
    new_samples = int(len(audio) * resample_factor)
    audio = scipy.signal.resample(audio, new_samples)
    return audio

def load_segment(full_path, target_sampling_rate=None, segment_size=None):

    if segment_size is not None:
        dur,sampling_rate,len_data = get_wave_duration(full_path)
        if sampling_rate is None: return None, None
        if sampling_rate < 44100: return None, None

        target_dur = segment_size / target_sampling_rate
        if dur < target_dur:
            data, sampling_rate = load_wav(full_path)
            #print(f'data_read: {data.shape}, sampling_rate: {sampling_rate}')
            if data is None: return None, None

            if target_sampling_rate is not None and sampling_rate != target_sampling_rate:
                data = resample(data, sampling_rate, target_sampling_rate)
                sampling_rate = target_sampling_rate
            data = torch.FloatTensor(data)
            data = data.unsqueeze(0)
            data = torch.nn.functional.pad(data, (0, segment_size - data.size(1)), 'constant')
            data = data.squeeze(0)
            return data.numpy(), sampling_rate
        else:
            dur,sampling_rate,len_data = get_wave_duration(full_path)
            if sampling_rate < 44100: return None, None
                        
            target_dur = segment_size / target_sampling_rate
            target_len = int(target_dur * sampling_rate)
            start_idx = random.randint(0, (len_data - target_len))
            start_ms = start_idx / sampling_rate * 1000
            end_ms = start_ms + target_dur * 1000
            data = read_audio_segment(full_path, start_ms, end_ms)    
            #print(f'data_read: {data.shape}, sampling_rate: {sampling_rate}')
            if data is None: return None, None
            if target_sampling_rate is not None and sampling_rate != target_sampling_rate:
                data = resample(data, sampling_rate, target_sampling_rate)
                sampling_rate = target_sampling_rate
            if len(data) < segment_size:
                data = torch.FloatTensor(data)
                data = data.unsqueeze(0)
                data = torch.nn.functional.pad(data, (0, segment_size - data.size(1)), 'constant')
                data = data.squeeze(0)
                data = data.numpy()
            else:
                start_idx = random.randint(0, (len(data) - segment_size))
                data = data[start_idx:start_idx+segment_size]
            #print(f'data_cut: {data.shape}')
            return data, sampling_rate
    else:
        dur,sampling_rate,len_data = get_wave_duration(full_path)
        if sampling_rate is None: return None, None
        if sampling_rate < 44100: return None, None
        data, sampling_rate = load_wav(full_path)
        if data is None: return None, None
        if target_sampling_rate is not None and sampling_rate != target_sampling_rate:
            data = resample(data, sampling_rate, target_sampling_rate)
            sampling_rate = target_sampling_rate
        return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    '''
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
    '''
    global mel_basis, hann_window
    if fmax not in mel_basis:
        #mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        # sr, n_fft, n_mels=128, fmin=0.0, fmax
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

class Wave_Processor(object):
    """
    A class for processing audio data, specifically for reading input and label audio files,
    segmenting them into fixed-length segments, and applying padding or trimming as necessary.

    Methods:
    process(path, segment_length, sampling_rate):
        Processes audio data by reading, padding, or segmenting it to match the specified segment length.
    
    Parameters:
    path (dict): A dictionary containing file paths for 'inputs' and 'labels' audio files.
    segment_length (int): The desired length of audio segments to extract.
    sampling_rate (int): The target sampling rate for reading the audio files.
    """

    def process(self, path, segment_length, sampling_rate):
        """
        Reads input and label audio files, and ensures the audio is segmented into
        the desired length, padding if necessary or extracting random segments if
        the audio is longer than the target segment length.

        Parameters:
        path (dict): Dictionary containing the paths to 'inputs' and 'labels' audio files.
        segment_length (int): Desired length of the audio segment in samples.
        sampling_rate (int): Target sample rate for the audio.

        Returns:
        tuple: A pair of numpy arrays representing the processed input and label audio,
               either padded to the segment length or trimmed.
        """
        # Read the input and label audio files using the target sampling rate.
        wave_inputs = audioread(path['inputs'], sampling_rate)
        wave_labels = audioread(path['labels'], sampling_rate)
        
        # Get the length of the label audio (assumed both inputs and labels have similar lengths).
        len_wav = wave_labels.shape[0]
        
        # If the input audio is shorter than the desired segment length, pad it with zeros.
        if wave_inputs.shape[0] < segment_length:
            # Create zero-padded arrays for inputs and labels.
            padded_inputs = np.zeros(segment_length, dtype=np.float32)
            padded_labels = np.zeros(segment_length, dtype=np.float32)
            
            # Copy the original audio into the padded arrays.
            padded_inputs[:wave_inputs.shape[0]] = wave_inputs
            padded_labels[:wave_labels.shape[0]] = wave_labels
        else:
            # Randomly select a start index for segmenting the audio if it's longer than the segment length.
            st_idx = random.randint(0, len_wav - segment_length)
            
            # Extract a segment of the desired length from the inputs and labels.
            padded_inputs = wave_inputs[st_idx:st_idx + segment_length]
            padded_labels = wave_labels[st_idx:st_idx + segment_length]
        
        # Return the processed (padded or segmented) input and label audio.
        return padded_inputs, padded_labels

class Fbank_Processor(object):
    """
    A class for processing input audio data into mel-filterbank (Fbank) features, 
    including the computation of delta and delta-delta features.
    
    Methods:
    process(inputs, args):
        Processes the raw audio input and returns the mel-filterbank features 
        along with delta and delta-delta features.
    """
    
    def process(self, inputs, args):
        # Convert frame length and shift from seconds to milliseconds.
        frame_length = int(args.win_len / args.sampling_rate * 1000)
        frame_shift = int(args.win_inc / args.sampling_rate * 1000)

        # Set up configuration for the mel-filterbank computation.
        fbank_config = {
            "dither": 1.0,
            "frame_length": frame_length,
            "frame_shift": frame_shift,
            "num_mel_bins": args.num_mels,
            "sample_frequency": args.sampling_rate,
            "window_type": args.win_type
        }

        # Convert the input audio to a FloatTensor and scale it to match the expected input range.
        inputs = torch.FloatTensor(inputs * MAX_WAV_VALUE_16B)

        # Compute the mel-filterbank features using Kaldi's fbank function.
        fbank = torchaudio.compliance.kaldi.fbank(inputs.unsqueeze(0), **fbank_config)

        # Add delta and delta-delta features.
        fbank_tr = torch.transpose(fbank, 0, 1)
        fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
        fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
        fbank_delta = torch.transpose(fbank_delta, 0, 1)
        fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)
        
        # Concatenate the original Fbank, delta, and delta-delta features.
        fbanks = torch.cat([fbank, fbank_delta, fbank_delta_delta], dim=1)
        
        return fbanks.numpy()

class Mel_Processor(object):

    def process(self, filename, args, audio_list):
        index = len(audio_list)
        while 1:
            #audio, sampling_rate = load_wav(filename)
            audio, sampling_rate = load_segment(filename, args.sampling_rate, args.segment_size)
            if audio is not None: break
            else:
                print(f'{filename} is excluded !')
                filename = audio_list[random.randint(0,index)]['inputs']

        sr_out = random.choice(args.supported_sampling_rates)
        audio_down = resample(audio, args.sampling_rate, sr_out)

        target_len = len(audio) 
        audio_input = resample(audio_down, None, None, target_len)

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        audio_input = torch.FloatTensor(audio_input)
        audio_input = audio_input.unsqueeze(0)
        input_mel = mel_spectrogram(audio_input, args.n_fft, args.num_mels,
                                  args.sampling_rate, args.hop_size, args.win_size, args.fmin, args.fmax,
                                  center=False)
        output_mel = mel_spectrogram(audio, args.n_fft, args.num_mels,
                                   args.sampling_rate, args.hop_size, args.win_size, args.fmin, args.fmax,
                                   center=False)
        return input_mel.squeeze().numpy(), audio.squeeze(0).numpy(), output_mel.squeeze().numpy()

class AudioDataset(Dataset):
    """
    A dataset class for loading and processing audio data from different data types 
    (train, validation, test). Supports audio processing and feature extraction 
    (e.g., waveform processing, Fbank feature extraction).

    Parameters:
    args: Arguments containing dataset configuration (paths, sampling rate, etc.).
    data_type (str): The type of data to load (train, val, test).
    """

    def __init__(self, args, data_type):
        self.args = args
        self.sampling_rate = args.sampling_rate
        
        # Read the list of audio files based on the data type.
        if data_type == 'train':
            self.wav_list = read_and_config_file(args, args.tr_list, decode=False)
        elif data_type == 'val':
            self.wav_list = read_and_config_file(args, args.cv_list, decode=False)
        elif data_type == 'test':
            self.wav_list = read_and_config_file(args, args.tt_list, decode=False)
        else:
            print(f'Data type: {data_type} is unknown!')
        
        # Initialize processors for waveform and Fbank features.
        #self.wav_processor = Wave_Processor()
        #self.fbank_processor = Fbank_Processor()
        self.mel_processor = Mel_Processor()
        # Clip data to a fixed segment length based on the sampling rate and max length.
        #self.segment_length = self.sampling_rate * self.args.max_length
        print(f'No. {data_type} files: {len(self.wav_list)}')

    def __len__(self):
        # Return the number of audio files in the dataset.
        return len(self.wav_list)

    def __getitem__(self, index):
        # Get the input and label paths from the list.
        data_info = self.wav_list[index]
        #print(f'index: {index}, data_info: {data_info}')
        '''
        # Process the waveform inputs and labels.
        inputs, labels = self.wav_processor.process(
            {'inputs': data_info['inputs'], 'labels': data_info['labels']}, 
            self.segment_length, 
            self.sampling_rate
        )
        
        # Optionally load Fbank features if specified.
        if self.args.load_fbank is not None:
            fbanks = self.fbank_processor.process(inputs, self.args)
            return inputs * MAX_WAV_VALUE_16B, labels * MAX_WAV_VALUE_16B, fbanks
        '''
        # Process the waveform inputs and extract mel features        
        input_mel, output_audio, output_mel = self.mel_processor.process(data_info['inputs'], self.args, self.wav_list)    
        return input_mel, output_audio, output_mel

def zero_pad_concat(self, inputs):
    """
    Concatenates a list of input arrays, applying zero-padding as needed to ensure 
    they all match the length of the longest input.

    Parameters:
    inputs (list of numpy arrays): List of input arrays to be concatenated.

    Returns:
    numpy.ndarray: A zero-padded array with concatenated inputs.
    """
    
    # Get the maximum length among all inputs.
    max_t = max(inp.shape[0] for inp in inputs)
    
    # Determine the shape of the output based on the input dimensions.
    shape = None
    if len(inputs[0].shape) == 1:
        shape = (len(inputs), max_t)
    elif len(inputs[0].shape) == 2:
        shape = (len(inputs), max_t, inputs[0].shape[1])
    
    # Initialize an array with zeros to hold the concatenated inputs.
    input_mat = np.zeros(shape, dtype=np.float32)
    
    # Copy the input data into the zero-padded array.
    for e, inp in enumerate(inputs):
        if len(inp.shape) == 1:
            input_mat[e, :inp.shape[0]] = inp
        elif len(inp.shape) == 2:
            input_mat[e, :inp.shape[0], :] = inp
    
    return input_mat

def collate_fn_2x_wavs(data):
    """
    A custom collate function for combining batches of waveform input and label pairs.

    Parameters:
    data (list): List of tuples (inputs, labels).

    Returns:
    tuple: Batched inputs and labels as torch.FloatTensors.
    """
    inputs, labels = zip(*data)
    x = torch.FloatTensor(inputs)
    y = torch.FloatTensor(labels)
    return x, y

def collate_fn_2x_wavs_fbank(data):
    """
    A custom collate function for combining batches of waveform inputs, labels, and Fbank features.

    Parameters:
    data (list): List of tuples (inputs, labels, fbanks).

    Returns:
    tuple: Batched inputs, labels, and Fbank features as torch.FloatTensors.
    """
    inputs, labels, fbanks = zip(*data)
    x = torch.FloatTensor(inputs)
    y = torch.FloatTensor(labels)
    z = torch.FloatTensor(fbanks)
    return x, y, z

def collate_fn_3x_mel(data):
    """
    A custom collate function for combining batches of waveform inputs, labels, and Fbank features.

    Parameters:
    data (list): List of tuples (inputs, labels, fbanks).

    Returns:
    tuple: Batched inputs, labels, and Fbank features as torch.FloatTensors.
    """
    inputs_mel, outputs_audio, outputs_mel = zip(*data)
    x = torch.FloatTensor(inputs_mel)
    y = torch.FloatTensor(outputs_audio)
    z = torch.FloatTensor(outputs_mel)
    return x, y, z #inputs_mel, outputs_audio, outputs_mel

class DistributedSampler(data.Sampler):
    """
    Sampler for distributed training. Divides the dataset among multiple replicas (processes), 
    ensuring that each process gets a unique subset of the data. It also supports shuffling 
    and managing epochs.

    Parameters:
    dataset (Dataset): The dataset to sample from.
    num_replicas (int): Number of processes participating in the training.
    rank (int): Rank of the current process.
    shuffle (bool): Whether to shuffle the data or not.
    seed (int): Random seed for reproducibility.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # Shuffle the indices based on the epoch and seed.
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ind = torch.randperm(int(len(self.dataset) / self.num_replicas), generator=g) * self.num_replicas
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind + i).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Add extra samples to make the dataset evenly divisible.
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample for the current process.
        indices = indices[self.rank * self.num_samples:(self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_dataloader(args, data_type):
    """
    Creates and returns a data loader and sampler for the specified dataset type (train, validation, or test).
    
    Parameters:
    args (Namespace): Configuration arguments containing details such as batch size, sampling rate, 
                      network type, and whether distributed training is used.
    data_type (str): The type of dataset to load ('train', 'val', 'test').
    
    Returns:
    sampler (DistributedSampler or None): The sampler for distributed training, or None if not used.
    generator (DataLoader): The PyTorch DataLoader for the specified dataset.
    """
    
    # Initialize the dataset based on the given arguments and dataset type (train, val, or test).
    datasets = AudioDataset(args=args, data_type=data_type)

    # Create a distributed sampler if distributed training is enabled; otherwise, use no sampler.
    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,  # Number of replicas in distributed training.
        rank=args.local_rank  # Rank of the current process.
    ) if args.distributed else None

    # Select the appropriate collate function based on the network type.
    if args.network == 'MossFormer2_SR_48K':
        # Use the collate function for waveforms along with Fbank features.
        collate_fn = collate_fn_3x_mel

    else:
        # Print an error message if the network type is unknown.
        print(f'in dataloader, please specify a correct network type using args.network!')
        return

    # Create a DataLoader with the specified dataset, batch size, and worker configuration.
    generator = data.DataLoader(
        datasets,
        batch_size=args.batch_size,  # Batch size for training.
        shuffle=(sampler is None),  # Shuffle the data only if no sampler is used.
        collate_fn=collate_fn,  # Use the selected collate function for batching data.
        num_workers=args.num_workers,  # Number of workers for data loading.
        sampler=sampler,  # Use the distributed sampler if applicable.
        pin_memory=True,
        drop_last=True
    )
    
    # Return both the sampler and DataLoader (generator).
    return sampler, generator
