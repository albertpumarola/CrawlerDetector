#!/usr/bin/python
import os
from tqdm import tqdm
import argparse
import glob
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('-ii', '--input_dir', type=str, default='/home/apumarola/datasets/aeroarms_dataset_processed/pos2/color', help='Input images directory')
parser.add_argument('-oi', '--output_dir', type=str, default='/home/apumarola/datasets/aeroarms_dataset_processed/pos2/color_selected', help='Output images directory')
parser.add_argument('-ih', '--header', type=str, default='vid1', help='Output header')
parser.add_argument('-n', '--every_n_frames', type=int, default=3, help='Select every n frames')
args = parser.parse_args()

def main():
    input_dir = glob.glob(os.path.join(args.input_dir, '*'))
    input_dir.sort()

    save_dir = args.output_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for img_dir in tqdm(input_dir[::args.every_n_frames]):
        save_filename = os.path.basename(img_dir)
        save_filename = os.path.join(save_dir, args.header + "_" + save_filename)
        copyfile(img_dir, save_filename)


if __name__ == '__main__':
    main()
