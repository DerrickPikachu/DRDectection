import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from parameter import *
from loop import train_model
from models import ResNet
from dataloader import RetinopathyLoader
from imgTransform import ImgToTorch

hyper_var = {
    'Adam': [0.001],
    'SGD': [0.001],
}


def train_diff_opti():
    file = open('past_record/record', 'w')
    # train_data = RetinopathyLoader('data', 'train', ImgToTorch())
    # test_data = RetinopathyLoader('data', 'test', ImgToTorch())
    #
    # loader = {'train': DataLoader(train_data, batch_size=batch_size, num_workers=4),
    #           'test': DataLoader(test_data, batch_size=batch_size, num_workers=4)}

    for model_type in ['pretrained', 'no_pretrained']:
        for opt, lr_list in hyper_var.items():
            for lr in lr_list:
                model = None
                if model_type == 'pretrained':
                    model = ResNet(resnet_depth=18, pretrained=True, feature_extracting=False)
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


def train_with_new_transform():
    train_config = {
        '18': [0.001],
        '50': [0.01, 0.001],
    }
    file = open('past_record/transform_record', 'w')

    for res_t, lr_list in train_config.items():
        for lr in lr_list:
            model = ResNet(resnet_depth=int(res_t), pretrained=True, feature_extracting=False)
            model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

            # Start training, use the parameter define in parameter.py
            model, record = train_model(model, loader, loss_fn, optimizer, epoch)

            # Save the data and record
            file.write(f'{record}\n')
            torch.save(model, f'{res_t}_{lr}.pth')

    file.close()


if __name__ == '__main__':
    train_with_new_transform()
