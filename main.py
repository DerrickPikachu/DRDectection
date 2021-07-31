import torchvision
from torchvision import transforms
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
    model = ResNet(50, True, False)
    model.to(device)
    print(model)

    # Init optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training start
    model, record = train_model(model, loader, loss_fn, optimizer, epoch)

    # Write the record into file
    file = open('single_model_record', 'w')
    file.write(f'{record}')
    file.close()

    # Save the trained model
    torch.save(model, 'model.pth')
