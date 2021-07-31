import torch
from torch.utils.data import DataLoader

from parameter import *
from loop import train_model
from models import ResNet
from dataloader import RetinopathyLoader
from imgTransform import ImgToTorch

hyper_var = {
    'Adam': [0.001],
    'SGD': [0.001],
}


if __name__ == '__main__':
    file = open('record', 'w')
    train_data = RetinopathyLoader('data', 'test', ImgToTorch())
    test_data = RetinopathyLoader('data', 'test', ImgToTorch())

    loader = {'train': DataLoader(train_data, batch_size=batch_size),
              'test': DataLoader(train_data, batch_size=batch_size)}

    for model_type in ['pretrained', 'no_pretrained']:
        for opt, lr_list in hyper_var.items():
            for lr in lr_list:
                model = None
                if model_type == 'pretrained':
                    model = ResNet(resnet_depth=18, pretrained=True, feature_extracting=True)
                else:
                    model = ResNet(resnet_depth=18, pretrained=False, feature_extracting=False)
                model.to(device)

                optimizer = None
                if opt == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

                model, record = train_model(model, loader, loss_fn, optimizer, epoch)
                # record = {
                #     'train': [1, 2, 3, 4, 5],
                #     'test': [5, 4, 3, 2, 1],
                # }
                file.write(f'type: {model_type}, opt: {opt}, lr: {lr}\n')
                file.write(f'{record}\n')
                torch.save(model, f'{model_type}_{opt}_{lr}.pth')

    file.close()
