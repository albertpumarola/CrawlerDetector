#!/usr/bin/python
import os
from tqdm import tqdm
import argparse
import glob
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('-ii', '--input_bags_dir', type=str, default='/home/lab/Documents/aeroarms_dataset/pos/color', help='Input images directory')
parser.add_argument('-oi', '--output_bags_dir', type=str, default='/home/lab/Documents/aeroarms_dataset_processed/pos/color', help='Output images directory')
parser.add_argument('-n', '--every_n_frames', type=int, default=5, help='Select every n frames')
args = parser.parse_args()

def main():
    input_bags_dir = glob.glob(os.path.join(args.input_bags_dir, '*'))

    save_dir = args.output_bags_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for bag_dir in tqdm(input_bags_dir):
        images_filenames = os.listdir(bag_dir)
        dir_name = bag_dir.split('/')[-1]
        images_filenames.sort()
        for i, image_filename in enumerate(images_filenames):
            if i % args.every_n_frames == 0:
                src_filename = os.path.join(bag_dir, os.path.basename(image_filename))
                save_filename = "%s_close_%s" % (dir_name, os.path.basename(image_filename))
                save_filename = os.path.join(save_dir, save_filename)
                copyfile(src_filename, save_filename)


if __name__ == '__main__':
    main()
