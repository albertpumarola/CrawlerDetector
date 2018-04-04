#!/usr/bin/env bash

GPU=0
export CUDA_VISIBLE_DEVICES=$GPU

python train.py \
--name debug \
--model object_detector_net_model_small \
--checkpoints_dir ./checkpoints \
--batch_size 20 \
--gpu_ids 0 \
--nepochs_no_decay 80 \
--nepochs_decay 20