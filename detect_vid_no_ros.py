# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
from api import CrawlerDetector
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vid_path', type=str, default="/home/apumarola/datasets/cagedetector/pos3/test.avi", help='video path')

args = parser.parse_args()

class DetectVidNoRos:
    def __init__(self, process_every_n=3):
        self._video_capture = cv2.VideoCapture(args.vid_path)
        self._crawler_detector = CrawlerDetector()
        self._process_every_n = process_every_n
        self._run()

    def _run(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 3.0, (224, 224))
        n_to_process = self._process_every_n
        i = 0
        while True:
            ret, frame = self._video_capture.read()
            n_to_process -= 1

            if ret and n_to_process == 0:
                _, _, result = self._detect_crawler(frame)
                out.write(result)
                n_to_process = self._process_every_n
                i += 1
            # Quit with q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if i == 40:
                break

        self._video_capture.release()
        out.release()
        cv2.destroyAllWindows()

    def _detect_crawler(self, frame):
        return self._crawler_detector.detect(frame, is_bgr=True, do_display_detection=False)

if __name__ == '__main__':
    DetectVidNoRos()
