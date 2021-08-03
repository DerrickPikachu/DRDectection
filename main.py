import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import torch

from parameter import *
from loop import train_model_pretrain, train_model
from dataloader import RetinopathyLoader
from imgTransform import ImgToTorch
from models import ResNet


if __name__ == '__main__':
    model = torch.load('model_80p.pth')
    # model = ResNet(18, True, False)
    model = model.to(device)
    print(model)

    # Init optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Training start
    model, record = train_model(model, loader, loss_fn, optimizer, epoch)
    # model, record = train_model_pretrain(model, loss_fn, optimizer, epoch)

    # Write the record into file
    file = open('rotation_record', 'w')
    file.write(f'{record}')
    file.close()

    # Save the trained model
    torch.save(model, 'model.pth')
