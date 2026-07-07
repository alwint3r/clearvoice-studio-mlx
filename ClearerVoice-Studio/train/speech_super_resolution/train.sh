#!/bin/sh

gpu_id=5				# visible GPUs
n_gpu=1				# number of GPU used for training
network=MossFormer2_SR_48K  #train which network
checkpoint_dir=checkpoints/$network						# leave empty if it's a new training, otherwise provide the 'log_name'
config_pth=config/train/${network}.yaml		# the config file, only used if it's a new training
train_from_last_checkpoint=1 #resume training from last checkpoint, 1 for true, 0 for false. If use 1 and last_checkpoint is not found, start a new training
print_freq=10  # No. steps waited for printing info
checkpoint_save_freq=500  #No. steps waited for saving new checkpoint

if [ ! -d "${checkpoint_dir}" ]; then
  mkdir -p ${checkpoint_dir}
fi

cp $config_pth $checkpoint_dir/config.yaml

export PYTHONWARNINGS="ignore"
CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=$n_gpu \
--master_port=$(date '+88%S') \
train.py \
--config ${config_pth} \
--checkpoint_dir ${checkpoint_dir} \
--train_from_last_checkpoint ${train_from_last_checkpoint} \
--init_checkpoint_path ${init_checkpoint_path} \
--print_freq ${print_freq} \
--checkpoint_save_freq ${checkpoint_save_freq}
