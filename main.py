import torchvision
from torch import nn
from torch.utils.data import DataLoader

from parameter import *
from loop import train_model
from dataloader import RetinopathyLoader
from imgTransform import ImgToTorch
from models import ResNet


if __name__ == '__main__':
    # model = torchvision.models.resnet50(pretrained=True)
    # model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=5, bias=True))
    # model.to(device)
    model = ResNet(18, True, False)
    model.to(device)
    print(model)

    train_data = RetinopathyLoader('data', 'train', transform=ImgToTorch())
    test_data = RetinopathyLoader('data', 'test', transform=ImgToTorch())

    loader = {}
    loader['train'] = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader['test'] = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_model(model, loader, loss_fn, optimizer, epoch)
    # print(model)
