import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from .resnet_backbone import get_backbone


class RandomInitialization(nn.Module):
    def __init__(self, num_classes, version='18'):
        super(RandomInitialization, self).__init__()
        self.backbone = get_backbone(out_features=num_classes, version=version)
        self.init_weight()

    def forward(self, x):
        return self.backbone(x)

    def init_weight(self):
        for name, m in self.backbone.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-5
                m.momentum = 0.1
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)