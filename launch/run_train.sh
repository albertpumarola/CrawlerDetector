#!/usr/bin/env bash

#GPU=0
#export CUDA_VISIBLE_DEVICES=$GPU

#python train.py \
#--name debug4 \
#--model object_detector_net_model_small \
#--checkpoints_dir ./checkpoints \
#--batch_size 30 \
#--gpu_ids 0 \
#--nepochs_no_decay 800 \
#--nepochs_decay 200

#python train.py \
#--name debug5 \
#--model object_detector_net_model_small \
#--checkpoints_dir ./checkpoints \
#--batch_size 30 \
#--gpu_ids 0 \
#--nepochs_no_decay 8000 \
#--nepochs_decay 2000


#GPU=1
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name debug6 \
#--model object_detector_net_model_small \
#--checkpoints_dir ./checkpoints \
#--batch_size 20 \
#--gpu_ids 0 \
#--nepochs_no_decay 8000 \
#--nepochs_decay 2000

#GPU=1
#export CUDA_VISIBLE_DEVICES=$GPU
#export CUDA_LAUNCH_BLOCKING=1
## no pretrained weights
#python train.py \
#--name debug11 \
#--model object_detector_net_model_small \
#--checkpoints_dir ./checkpoints \
#--batch_size 150 \
#--gpu_ids 0 \
#--nepochs_no_decay 10000 \
#--nepochs_decay 50

GPU=1
export CUDA_VISIBLE_DEVICES=$GPU
# no pretrained weights
python train.py \
--name prob_map7 \
--model object_detector_net_prob_map \
--checkpoints_dir ./checkpoints \
--batch_size 120 \
--gpu_ids 0 \
--poses_g_sigma 1.2 \
--nepochs_no_decay 80 \
--nepochs_decay 20