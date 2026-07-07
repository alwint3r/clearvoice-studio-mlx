import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir
import os, sys
import random
import yamlargparse

def main(args):
    num_RIRs = args.num_RIRs
    curr_run = 'run'+args.run_num
    out_dir_curr = args.output_path+'/'+curr_run+'/'+args.output_RIR_dir
    if not os.path.exists(out_dir_curr):
        cmd = 'mkdir -p '+out_dir_curr
        os.system(cmd)

    wav_list=out_dir_curr+'/wav.lst'
    f = open(wav_list, 'w')

    fs = args.sample_rate
    for i in range(num_RIRs):
        print('count: {}'.format(i))
        RT = round(np.clip(random.gauss(0,0.1)+0.6, 0.2, 1.0),2)
        print('Reverberation time:  {}'.format(RT))
        rm_x = round(random.uniform(7,11),2)
        rm_y = round(random.uniform(5,7),2)
        rm_z = round(random.uniform(3.4,3.6),2)
        print('room size: x={},y={},z={}'.format(rm_x,rm_y,rm_z))
        mic_x = round(random.uniform(1,rm_x-1),2)
        mic_y = round(random.uniform(1,rm_y-1),2)
        mic_z = round(random.uniform(2,3),2)
        print('mic pos: x={},y={},z={}'.format(mic_x,mic_y,mic_z))
        src_x = round(random.uniform(0.3,rm_x-0.3),2)
        src_y = round(random.uniform(0.3,rm_y-0.3),2)
        src_z = round(random.uniform(0.8,1.8),2)
        print('src pos: x={},y={},z={}'.format(src_x,src_y,src_z))
        h = rir.generate(
            c=340,                  # Sound velocity (m/s)
            fs=fs,                  # Sample frequency (samples/s)
            r=[mic_x, mic_z, mic_y],                     # Receiver position(s) [x y z] (m)
            s=[src_x, src_z, src_y],          # Source position [x y z] (m)
            L=[rm_x, rm_z, rm_y],            # Room dimensions [x y z] (m)
            reverberation_time=RT, # Reverberation time (s)
            nsample=int(fs * RT),           # Number of output samples
        )

        print(h.shape)              # (4096, 3)
        out_path_rir = out_dir_curr+'/rir_'+curr_run+'_'+str(i)+'_RT_'+str(RT)+'.wav'
        f.write(out_path_rir + '\n')
        sf.write(out_path_rir, h, fs)
    f.close()

if __name__ == '__main__':
    parser = yamlargparse.ArgumentParser("Settings")
    # experiment setting
    parser.add_argument('--output_path', type=str, default='./data', help='output will be save to this path')
    parser.add_argument('--output_RIR_dir', type=str, default='output_RIRs', help='output will be save to this dir')
    parser.add_argument('--run_num', type=str, default='0', help='set to 0,1,2,3,... for different run')
    parser.add_argument('--num_RIRs', type=int, default=10, help='number of RIRs to be generated')
    parser.add_argument('--sample_rate', type=int, default=48000, help='sampling rate for generated audio')
    
    args, _ = parser.parse_known_args()
    print(args)
    main(args)
