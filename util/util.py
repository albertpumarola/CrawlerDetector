from __future__ import print_function
from PIL import Image
import numpy as np
import os
import torchvision
import cv2
import math


def tensor2im(img, imtype=np.uint8, unnormalize=True, nrows=None, to_numpy=False, scale=1, interpolation=cv2.INTER_LINEAR):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = torchvision.utils.make_grid(img, nrow=nrows)

    # unnormalize
    img = img.cpu().float()
    if unnormalize:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)
        img *= 255

    # to numpy
    image_numpy = img.numpy()
    image_numpy = resize_numpy_tensor(image_numpy, scale, interpolation)
    if to_numpy:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy.astype(imtype)

def resize_numpy_tensor(tensor, scale, interpolation):
    tensor = tensor.transpose((1, 2, 0))
    tensor = cv2.resize(tensor, None, fx=scale, fy=scale, interpolation=interpolation)
    tensor = tensor.transpose((2, 0, 1))
    return tensor


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)