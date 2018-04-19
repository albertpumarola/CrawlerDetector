import cv2
import datetime
import torch
import torchvision.transforms as transforms
from options.test_options import TestOptions
from models.models import ModelsFactory
import numpy as np

class CrawlerDetector:
    def __init__(self):
        self._opt = TestOptions().parse()  # parse model parameters
        self._img2tensor = self._create_img_transform()  # map RGB cv2 image to Pytorch tensor
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)  # get model
        self._model.set_eval()  # set model in test mode

    def detect(self, frame, is_bgr=False, do_display_detection=False):
        # detect crawler
        frame_tensor = self._preprocess_frame(frame, is_bgr)
        bb, prob, elapsed_time = self._detect_crawler(frame_tensor)
        bb = self._restore_bb_size(bb, frame.shape)

        # display detection
        if do_display_detection:
            self._display_detection(frame, bb, prob, elapsed_time)

        return bb, prob

    def _restore_bb_size(self, bb, frame_size):
        h, w, _ = frame_size
        top, left, bottom, right = bb
        h_ratio = h / float(self._opt.image_size_h)
        w_ratio = w / float(self._opt.image_size_w)
        return int(top*h_ratio), int(left*w_ratio), int(bottom*h_ratio), int(right*w_ratio)

    def _preprocess_frame(self, frame, is_bgr):
        # resize frame to network input size
        frame = cv2.resize(frame, (self._opt.image_size_h, self._opt.image_size_w))

        # if necessary BGR to RGB
        if is_bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # convert to pytorch tensor and add batch dimension
        frame_tensor = torch.unsqueeze(self._img2tensor(frame), 0)

        return frame_tensor

    def _detect_crawler(self, frame_tensor):
        # bb as (top, left, bottom, right)
        start_time = datetime.datetime.now()
        bb, prob = self._model.test(frame_tensor)
        # bb = np.array([[93, 99, 235, 266]])
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        return bb[0], prob[0], elapsed_time

    def _display_detection(self, frame, bb, prob, elapsed_time):
        color = self._get_display_color(self._is_pos_detection(prob))

        # display bb
        (top, left, bottom, right) = bb
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # display prob
        prob_txt = '%.2f' % prob
        cv2.rectangle(frame, (left, bottom - 17), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, prob_txt, (left + 6, bottom - 3), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # display detection time
        h, w, _ = frame.shape
        detection_time_txt = 'Detection Time[s]: %.3f' % elapsed_time
        cv2.putText(frame, detection_time_txt, (w-200, h-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,0), 1)

        # display frame
        cv2.imshow('Crawler Detector', frame)

    def _create_img_transform(self):
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                               std=[0.5, 0.5, 0.5])]
        return transforms.Compose(transform_list)

    def _is_pos_detection(self, prob):
        return prob >= self._opt.classifier_threshold

    def _get_display_color(self, is_pos=True):
        return (0, 255, 0) if is_pos else (0, 0, 255)


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    crawler_detector = CrawlerDetector()
    while True:
        # capture frame
        ret, frame = video_capture.read()

        # if frame correctly captured detect crawler
        if ret:
            bb, prob = crawler_detector.detect(frame, is_bgr=True, do_display_detection=True)

        # Quit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle
    video_capture.release()
    cv2.destroyAllWindows()