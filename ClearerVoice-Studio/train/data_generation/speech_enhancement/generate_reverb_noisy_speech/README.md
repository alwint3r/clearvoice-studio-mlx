This script generates training data for speech enhancement models. Once the data generation is complete, you can use the output to retrain or fine-tune your models.

# Data Mixing Process
Data Mixing Process generates reverb&noisy-clean-speech pairs by mixing reverberant speech with additive background noise. The mixing process follows these steps:
1. Room Impulse Response (RIR) Generation: A randomlhy selected room size is used to create RIRs between a microphone and a speaker source location. This process is repeated many times.
2. Clean File Selection: For each RIR, several clean speech files can be selected for generating reverberant speech files.
3. SNR Selection: A random signal-to-noise ratio (SNR) within a predefined range is chosen for each mixture.
4. Noise Sampling: For each reverberant speech signal, a noise file is randomly selected.
5. Length Matching: If the noise signal is shorter than the clean speech, additional noise segments are concatenated to match the duration.
6. Target Speech Generation: It is not recommended to use the orignal clean speech as target. It is because the convolution process produces a time delay which generates a time shift of the clean speech signal. Therefore, the reverberant signal and the clean signal is not aligned any more. In addition, their lengths also don't match. We propose to keep early reverberations in the target speech. That is, we create the target speech as a reverberant speech with only early reverberations (the early reverberation length can be adjusted, by default = 0.1s). 

# Usage Instructions
1. Prepare file lists: Provide two file lists: one for clean speech and another for noise. Examples can be found in `data/data_scp/speech.scp` and `data/data_scp/noise.scp`.
2. Configure settings: Adjust the parameters in `config/para.cfg` according to your needs. To test the script, run the demo using `bash run.sh`.
3. Loop generation (optional): Set run_num in `run.sh` to repeat the generation process for multiple outputs.
