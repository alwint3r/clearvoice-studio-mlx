#!/bin/bash 

#use MossFormer2_SR_48K for speech super-resolution
network=MossFormer2_SR_48K

config=config/inference/${network}.yaml
CUDA_VISIBLE_DEVICES=0 python3 -u inference.py --config $config
