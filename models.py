import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable
import time
import functools

import models


class ResNet(nn.Module):
    def __init__(self, resnet_depth=50, pretrained=True, feature_extracting=True):
        super(ResNet, self).__init__()

        classifier = None
        self.resnet = None
        if resnet_depth == 18:
            self.resnet = torchvision.models.resnet18(pretrained)
            classifier = nn.Sequential(
                nn.Linear(in_features=512, out_features=5, bias=True),
                nn.Dropout(p=0.5)
            )
        elif resnet_depth == 50:
            self.resnet = torchvision.models.resnet50(pretrained)
            classifier = nn.Sequential(nn.Linear(in_features=2048, out_features=5, bias=True))

        self.setup_feature_extract(feature_extracting)
        self.resnet.fc = classifier

    def setup_feature_extract(self, feature_extracting):
        if feature_extracting:
            for parameter in self.resnet.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        return x

    # def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
    #     self.resnet.load_state_dict(state_dict=state_dict, strict=strict)

# class ResNet(nn.Module):
#     def __init__(self, pretrained=True):
#         super(ResNet, self).__init__()
#
#         self.classify = nn.Linear(2048, 5)
#
#         pretrained_model = torchvision.models.__dict__['resnet{}'.format(50)](pretrained=False)
#         self.conv1 = pretrained_model._modules['conv1']
#         self.bn1 = pretrained_model._modules['bn1']
#         self.relu = pretrained_model._modules['relu']
#         self.maxpool = pretrained_model._modules['maxpool']
#
#         self.layer1 = pretrained_model._modules['layer1']
#         self.layer2 = pretrained_model._modules['layer2']
#         self.layer3 = pretrained_model._modules['layer3']
#         self.layer4 = pretrained_model._modules['layer4']
#
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#
#         del pretrained_model
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         # print(x.shape)
#         x = x.view(x.size(0), -1)
#         x = self.classify(x)
#
#         return x
