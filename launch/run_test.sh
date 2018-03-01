#!/usr/bin/env bash

python test.py \
--test_data_dir ./datasets/Single-Object-Detector-Sample-Dataset/test/ \
--name test_aeroplanes \
--checkpoints_dir ./checkpoints/ \
--output_estimation_dir ./checkpoints/ \
--batch_size 23 \
--gpu_ids 0,1