This script generates training data for speech enhancement models. Once the data generation is complete, you can use the output to retrain or fine-tune your models.

# Data Mixing Process
Data Mixing Process generates noisy-clean-speech pairs by mixing clean speech with additive background noise. The mixing process follows these steps:
1. SNR Selection: A random signal-to-noise ratio (SNR) within a predefined range is chosen for each mixture.
2. Noise Sampling: For each clean speech signal, a noise file is randomly selected.
3. Length Matching: If the noise signal is shorter than the clean speech, additional noise segments are concatenated to match the duration.

# Usage Instructions
1. Prepare file lists: Provide two file lists: one for clean speech and another for noise. Examples can be found in `data/data_scp/speech.scp` and `data/data_scp/noise.scp`.
2. Configure settings: Adjust the parameters in `config/para.cfg` according to your needs. To test the script, run the demo using `bash run.sh`.
3. Loop generation (optional): Set run_num in `run.sh` to repeat the generation process for multiple outputs.
