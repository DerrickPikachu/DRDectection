import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import RetinopathyLoader, PretrainLoader, EvenLoader
from imgTransform import ImgToTorch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test
# epoch = 1

epoch = 20
batch_size = 2
learning_rate = 0.001
loss_fn = torch.nn.CrossEntropyLoss()

# TODO: Need to survey the effect of these parameters
momentum = 0.9
weight_decay = 5e-4


pretrain_transform = transforms.Compose([
    ImgToTorch(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
pretest_transform = transforms.Compose([
    ImgToTorch(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_transform = transforms.Compose([
    ImgToTorch(),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomCrop(224),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    ImgToTorch(),
    transforms.Resize(224),
    # transforms.CenterCrop(224),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test
# train_data = RetinopathyLoader('data', 'test', transform=train_transform)

pretrain_data = PretrainLoader('data', 'train', transform=pretrain_transform)
pretest_data = PretrainLoader('data', 'test', transform=pretest_transform)
# train_data = RetinopathyLoader('data', 'train', transform=train_transform)
# test_data = RetinopathyLoader('data', 'test', transform=test_transform)
train_data = RetinopathyLoader('data', 'train', transform=train_transform)
test_data = RetinopathyLoader('data', 'test', transform=test_transform)

loader = {'train': DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4),
          'test': DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)}
pretrain_loader = {'train': DataLoader(pretrain_data, batch_size=batch_size, shuffle=True, num_workers=4),
                   'test': DataLoader(pretest_data, batch_size=batch_size, shuffle=True, num_workers=4)}
