import numpy as np
import scipy.signal as ss
import soundfile as sf
import os
import random
import yamlargparse

def perform_convolution(h, h_trim, clean_filelist, num_clean_files, out_path_signal_reverb, out_path_signal_target, target_sample_rate):
    clean_file_idx = random.randint(0, num_clean_files-1)
    clean_file = clean_filelist[clean_file_idx].strip()
    print('wav file: {}'.format(clean_file))
    signal, fs = sf.read(clean_file, always_2d=True)
    signal = signal[:,0:1]
    assert fs == target_sample_rate, "Signal sampling rate is not equal to {}".format(target_sample_rate)
    print('h: {}'.format(h.shape))              # (4096, 1)
    print('signal: {}'.format(signal.shape))         # (11462, 1)

    # Convolve 1-channel signal with 2 impulse responses
    signal_reverb = ss.convolve(h[:, None, :], signal[:, :, None])
    signal_target = ss.convolve(h_trim[:, None, :], signal[:, :, None])
    norm_term = max([max(abs(signal_reverb)), max(abs(signal_target))])
    signal_reverb = signal_reverb / norm_term * 0.9
    signal_target = signal_target / norm_term * 0.9

    print('signal_reverb: {}'.format(signal_reverb.shape)) 
    print('signal_target: {}'.format(signal_target.shape))
    print('out_path_signal_reverb: {}'.format(out_path_signal_reverb))
    print('out_path_signal_target: {}'.format(out_path_signal_target))
    sf.write(out_path_signal_reverb, signal_reverb[:,:,0], fs)
    sf.write(out_path_signal_target, signal_target[:,:,0], fs)

def main(args):
    curr_run = 'run'+args.run_num
    out_dir_curr = args.output_path+'/'+curr_run

    RIRs_list= out_dir_curr+'/'+args.output_RIR_dir+'/wav.lst'
    f_rir = open(RIRs_list,'r')
    f_wav = open(args.speech_scp,'r')
    clean_filelist = f_wav.readlines()
    num_clean_files = len(clean_filelist)
    print('num_clean_files: {}'.format(num_clean_files))
    f_wav.close

    out_dir = out_dir_curr + '/' + args.output_reverb_dir
    out_dir_signal_reverb=out_dir + '/reverb'
    out_dir_signal_target=out_dir + '/target'
    out_wav_list = out_dir + '/wav.lst'

    if not os.path.exists(out_dir_signal_reverb):
        cmd = 'mkdir -p '+out_dir_signal_reverb
        os.system(cmd)
        cmd = 'mkdir -p '+out_dir_signal_target
        os.system(cmd)

    f_wav_list = open(out_wav_list, 'w')
    num_wavs_per_RIR = args.num_wavs_per_RIR
    h_trim_point = int(args.sample_rate*args.target_rt)

    count = 0
    while 1:
        rir_path = f_rir.readline().strip()
        if not rir_path: break
        count = count + 1
        rir_name = rir_path.split('/')[-1].replace('.wav','')
        h, fs = sf.read(rir_path, always_2d=True)
        h_trim, fs = sf.read(rir_path, always_2d=True)
        assert fs == args.sample_rate, "RIR sampling rate is not equal to {}".format(args.sample_rate)
        h_trim[h_trim_point:,:] = 0
        for j in range(num_wavs_per_RIR):
            out_wav_name = 'sig_'+curr_run+'_'+str(count)+'_'+rir_name+'.wav'
            f_wav_list.write(out_wav_name+'\n')
            out_path_signal_reverb = out_dir_signal_reverb+'/'+ out_wav_name
            out_path_signal_target = out_dir_signal_target+'/'+ out_wav_name
            perform_convolution(h, h_trim, clean_filelist, num_clean_files, out_path_signal_reverb, out_path_signal_target, args.sample_rate)
            count = count + 1    
    f_rir.close()
    f_wav_list.close()

if __name__ == '__main__':
    parser = yamlargparse.ArgumentParser("Settings")
    # experiment setting
    parser.add_argument('--output_path', type=str, default='./data', help='output will be save to this path')
    parser.add_argument('--output_RIR_dir', type=str, default='output_RIRs', help='dir for storing RIRs')
    parser.add_argument('--output_reverb_dir', type=str, default='output_reverb_speech', help='output reverb speech will be save to this dir')
    parser.add_argument('--speech_scp', type=str, default='./data/data_scp/speech.scp', help='scp file to storing speech wav list')
    parser.add_argument('--run_num', type=str, default='0', help='set to 0,1,2,3,... for different run')
    parser.add_argument('--sample_rate', type=int, default=48000, help='sampling rate for generated audio')
    parser.add_argument('--num_wavs_per_RIR', type=int, default=3, help='how many utterances to be gen using one RIR')
    parser.add_argument('--target_rt', type=float, default=0.1, help='100ms, keep the early reverb as target speech')
    args, _ = parser.parse_known_args()
    print(args)
    main(args)
