# ClearerVoice-Studio: Train Speech Super-Resolution Models

## 1. Introduction

This repository provides a flexible training and finetune scripts for speech super-resolution. It supports both the pre-trained model `MossFormer2_SR_48K` or your customized model. The pre-trained model `MossFormer2_SR_48K` currently scales multiple lower sampling rates (>= 16kHz) to 48kHz sampling rate.

**MossFormer2_SR_48K**

`MossFormer2_SR_48K` is a speech super-resolution model with a target sampling rate of 48000 Hz. It aims to reconstruct a high-resolution speech signal from a low-resolution input that retains only a portion of the original samples. Consequently, `MossFormer2_SR_48K` delivers not only a superior listening experience but also improves speech intelligibility.

`MossFormer2_SR_48K` features a unified transformer-convolutional generator that seamlessly handles both the prediction of latent representations and their conversion into time-domain waveforms. This design allows the latent representations to move beyond mel-spectrogram constraints, enabling the transformer network to optimize them for optimal alignment with the convolutional network during waveform generation.

## 2. Usage

### Step-by-Step Guide

If you haven't created a Conda environment for ClearerVoice-Studio yet, follow steps 1 and 2. Otherwise, skip directly to step 3.

1. **Clone the Repository**

``` sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
```

2. **Create Conda Environment**

``` sh
cd ClearerVoice-Studio
conda create -n ClearerVoice-Studio python=3.8
conda activate ClearerVoice-Studio
pip install -r requirements.txt
```
Notice: Other higher python versions such as python 3.12.1 should also be supported.

3. **Prepare Dataset**
   
If you don't have any training dataset to start with, we recommend you to download the VCTK corpus ([link](https://datashare.ed.ac.uk/handle/10283/3443)]. You may store the dataset anywhere. What you need to start the model training is to create two scp files as shown in `train.scp` and `cv.scp`. `train.scp` contains the full path list of 48 kHz clean speech for training and `cv.scp` contains the full path list of 48 kHz clean speech for validation.

in train.scp：

`path_to_train_file1.wav`

`path_to_train_file2.wav`

`...`

in cv.scp：

`path_to_cv_file1.wav`

`path_to_cv_file2.wav`

`...`

Place scp files to `data/train.scp` and `data/cv.scp` and modify the scp paths in `config/train/*.yaml`. Now it is ready to train the models.

4. **Start Training**

``` sh
bash train.sh
```

You may need to set the correct network in `train.sh` and choose either a fresh training or a finetune process using:
```
network=MossFormer2_SR_48K            #Train MossFormer2_SR_48K model
train_from_last_checkpoint=1          #Set 1 to start training from the last checkpoint if exists, otherwise, set it to 0
```

> **Note:** We provide pretrained model checkpoints at HuggingFace ([link](https://huggingface.co/alibabasglab/MossFormer2_SR_48K/tree/main)). You may downlaod the whole model dir `MossFormer2_SR_48K` and place it to `./checkpoints/`
