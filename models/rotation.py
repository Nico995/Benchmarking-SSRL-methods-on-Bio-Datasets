import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from .resnet_backbone import get_backbone


class Rotation(nn.Module):
    def __init__(self, num_classes=4, version='18', weights=None):
        super(Rotation, self).__init__()
        # Out_features are set to 4 because we have 4 possible rotations to predict
        self.backbone = get_backbone(out_features=num_classes, version=version, weights=weights)

    def forward(self, x):
        return self.backbone(x)
