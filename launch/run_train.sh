#!/usr/bin/env bash

dataset_dir=./datasets/Single-Object-Detector-Sample-Dataset/

if [ ! -d $dataset_dir ];then
    URL=https://www.dropbox.com/s/nv3wn58nj6f8127/Single-Object-Detector-Sample-Dataset.zip?dl=0
    ZIP_FILE=./datasets/sample_dataset.zip
    TARGET_DIR=./datasets/
    mkdir $TARGET_DIR
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d $TARGET_DIR
    rm $ZIP_FILE
fi

python train.py \
--train_data_dir $dataset_dir"train/" \
--test_data_dir $dataset_dir"test/" \
--name aeroplanes \
--checkpoints_dir ./checkpoints \
--batch_size 23 \
--gpu_ids 0,1 \
--nepochs_no_decay 30 \
--nepochs_decay 30