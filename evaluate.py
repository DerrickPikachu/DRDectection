import torch
from torch.utils.data import DataLoader

from models import ResNet
from dataloader import RetinopathyLoader
from imgTransform import ImgToTorch
from parameter import *

hyper_var = {
    'Adam': [0.01, 0.001],
    'SGD': [0.01, 0.001],
}

if __name__ == "__main__":
    test_data = RetinopathyLoader('data', 'test', ImgToTorch)
    test_loader = DataLoader(test_data, batch_size=batch_size)
