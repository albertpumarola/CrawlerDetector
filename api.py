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
        crop_frame, frame_tensor = self._preprocess_frame(frame, is_bgr)
        hm, uv_max, prob, elapsed_time = self._detect_crawler(frame_tensor)

        # display detection
        if do_display_detection:
            self._display_hm(crop_frame, hm, uv_max, prob, elapsed_time)
        return hm, uv_max

    def _preprocess_frame(self, frame, is_bgr):
        # resize frame to half
        frame = cv2.resize(frame, (self._opt.image_size_w, self._opt.image_size_h))

        # crop frame to network size
        crop_top = int((self._opt.image_size_h-self._opt.net_image_size)/2.0)
        crop_left = int((self._opt.image_size_w - self._opt.net_image_size) / 2.0)
        crop_frame = frame[crop_top:crop_top+self._opt.net_image_size, crop_left:crop_left+self._opt.net_image_size]

        # convert to pytorch tensor and add batch dimension
        frame_tensor = crop_frame.copy()
        if is_bgr:
            frame_tensor = cv2.cvtColor(frame_tensor, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.unsqueeze(self._img2tensor(frame_tensor), 0)

        return crop_frame, frame_tensor

    def _detect_crawler(self, frame_tensor):
        # bb as (top, left, bottom, right)
        start_time = datetime.datetime.now()
        hm, uv_max, prob = self._model.test(frame_tensor, do_normalize_output=False)
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        return hm[0], uv_max, prob, elapsed_time

    def _display_bb(self, frame, bb, prob, elapsed_time):
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
        cv2.imshow('Crawler Detector BB', frame)

    def _display_hm(self, frame, hm, uv_max, prob, elapsed_time, prob_threshold=0.5):
        # display hm
        hm = (np.transpose(hm, (1, 2, 0)) * 255).astype(np.uint8)
        if prob is not None and prob < prob_threshold:
            hm = np.ones(hm.shape, dtype=hm.dtype)
        hm_img = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.7, hm_img, 0.3, 0)

        #display uv_max
        # frame = cv2.circle(frame, uv_max, 3, (255, 255, 0))

        # display detection time
        h, w, _ = frame.shape
        detection_time_txt = 'Detection Time[s]: %.3f' % elapsed_time
        cv2.putText(frame, detection_time_txt, (w - 200, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)

        # prob
        cv2.putText(frame, '%.2f' % prob, (w - 200, h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)

        # display frame
        cv2.imshow('Crawler Detector HM', frame)

    def _create_img_transform(self):
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                               std=[0.5, 0.5, 0.5])]
        return transforms.Compose(transform_list)

    def _is_pos_detection(self, prob):
        return prob >= self._opt.classifier_threshold

    def _get_display_color(self, is_pos=True):
        return (0, 255, 0) if is_pos else (0, 0, 255)
