output_path=./data # output data to be saved to this dir
run_num=1 #set 0, 1, 2, ... for different runs
num_RIRs=10 #number of RIRs to be generated
sample_rate=48000 

## Step 1: generate room impulse response (RIR) and save to wav files
python step1_gen_RIRs.py \
  --output_path $output_path \
  --run_num $run_num \
  --num_RIRs $num_RIRs \
  --sample_rate $sample_rate 

## Step 2: generate reverberant speech from clean speech by applying RIRs
python step2_gen_reverb_speech.py \
  --output_path $output_path \
  --run_num $run_num \
  --sample_rate $sample_rate

## Step3: add background noise to reverberant speech
python step3_add_noise.py \
  --output_path $output_path \
  --run_num $run_num \
  --sample_rate $sample_rate
