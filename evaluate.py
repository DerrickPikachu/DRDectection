import torch
from torch.utils.data import DataLoader

from models import ResNet
from dataloader import RetinopathyLoader
from imgTransform import ImgToTorch
from parameter import *

hyper_var = {
    'Adam': [0.001],
    'SGD': [0.001],
}

def evaluate(model, dataloader, loss, optimizer, epochs):


if __name__ == "__main__":
    test_data = RetinopathyLoader('data', 'test', ImgToTorch)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    for model_type in ['pretrained', 'no_pretrained']:
        for opt, lr_list in hyper_var.items():
            for lr in lr_list:
                filename = f'{model_type}_{opt}_{lr}.pth'
                model = torch.load(filename)

                if opt == 'Adam':
                    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                else:
