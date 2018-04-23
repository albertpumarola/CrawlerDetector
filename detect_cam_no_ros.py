import cv2
from api import CrawlerDetector

if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    crawler_detector = CrawlerDetector()
    while True:
        # capture frame
        ret, frame = video_capture.read()

        # frame = cv2.imread('/home/apumarola/datasets/Dataset-CrawlerDetection/selected_hm/pos/imgs/1519393652.030529.png')

        # if frame correctly captured detect crawler
        if ret:
            hm, uv_max = crawler_detector.detect(frame, is_bgr=True, do_display_detection=True)

        # Quit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle
    video_capture.release()
    cv2.destroyAllWindows()