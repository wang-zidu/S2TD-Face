import numpy as np
from scipy.io import loadmat
from PIL import Image
try:
    from PIL.Image import Resampling
    RESAMPLING_METHOD = Resampling.BICUBIC
except ImportError:
    from PIL.Image import BICUBIC
    RESAMPLING_METHOD = BICUBIC

import cv2
import os
from skimage import transform as trans
import torch
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)


def sketch_tensor(im, to_tensor=True):

    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return im

def get_data_path(root):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]

    return im_path

def make_square(image, fill_color=(255, 255, 255)):

    width, height = image.size

    fill_width = max(width, height)
    fill_height = max(width, height)

    new_image = Image.new('RGB', (fill_width, fill_height), fill_color)

    offset = ((fill_width - width) // 2, (fill_height - height) // 2)
    new_image.paste(image, offset)
    return new_image

def resize_image(image, size):
    return image.resize(size)