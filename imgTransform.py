import numpy as np
import torch


class ImgToTorch(object):
    def __init__(self):
        pass

    def __call__(self, img: np.ndarray):
        # Transpose to C, H, W because img in numpy is H, W, C, but img in torch should be C, H, W
        new_img = img.transpose((2, 0, 1)).astype('float')
        # Normalize the pixel value to [0, 1], because the value between [0, 1] will optimize gradient decent and become
        # more accurate
        new_img /= 255
        return torch.from_numpy(new_img)
