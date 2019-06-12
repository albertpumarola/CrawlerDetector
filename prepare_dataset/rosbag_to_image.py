#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import os
import argparse
import glob
import cv2
from tqdm import tqdm

#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file_dir", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()


    bag_files = glob.glob(args.bag_file_dir + "*.bag")
    bag_files.sort()
    i = 0
    for bag_file in tqdm(bag_files):
        bag = rosbag.Bag(bag_file, "r")
        bridge = CvBridge()
        count = 0
        print("start " + bag_file)
        out_dir = os.path.join(args.output_dir, "seq%02i" % i)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        for topic, msg, t in bag.read_messages(topics=[args.image_topic]):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv_img = cv_img[..., ::-1]
            cv2.imwrite(os.path.join(out_dir, "frame%06i.png" % count), cv_img)
            #print ("Wrote image %i" % count)

            count += 1

        bag.close()
        i+=1

    return

if __name__ == '__main__':
    main()
