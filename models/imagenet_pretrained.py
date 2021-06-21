import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from .resnet_backbone import get_backbone


class ImagenetPretrained(nn.Module):
    def __init__(self, num_classes, version='18'):
        super(ImagenetPretrained, self).__init__()
        self.backbone = get_backbone(out_features=num_classes, version=version, pretrained=True)

    def forward(self, x):
        return self.backbone(x)
