import cv2
from api import CrawlerDetector
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('-v', '--vid_path', type=str, default="/home/apumarola/datasets/aeroarms_dataset_processed/pos/color/output.mp4", help='video path')
# parser.add_argument('-v', '--vid_path', type=str, default="/home/apumarola/datasets/aeroarms_dataset_processed/pos/color/output.mp4", help='video path')
parser.add_argument('-v', '--vid_path', type=str, default="/home/apumarola/datasets/aeroarms_dataset_processed/pos/vid1.mp4", help='video path')

args = parser.parse_args()

class DetectVidNoRos:
    def __init__(self, process_every_n=3):
        self._video_capture = cv2.VideoCapture(args.vid_path)
        self._crawler_detector = CrawlerDetector()
        self._process_every_n = process_every_n
        self._run()

    def _run(self):
        n_to_process = self._process_every_n

        while True:
            ret, frame = self._video_capture.read()
            n_to_process -= 1

            if ret and n_to_process == 0:
                self._detect_crawler(frame)
                n_to_process = self._process_every_n

            # Quit with q
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        self._video_capture.release()
        cv2.destroyAllWindows()

    def _detect_crawler(self, frame):
        return self._crawler_detector.detect(frame, is_bgr=True, do_display_detection=True)

if __name__ == '__main__':
    DetectVidNoRos()