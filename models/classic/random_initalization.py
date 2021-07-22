import torch
from torch import nn
from torch.nn import Sequential, Linear, Conv2d, BatchNorm2d

from models.resnet_backbone import get_backbone


class RandomInitialization(nn.Module):
    def __init__(self, num_classes, version='18', weights=None, mode="pretext"):
        super(RandomInitialization, self).__init__()
        self.backbone, self.backbone_features = get_backbone(version=version, weights=weights)
        self.classifier = Sequential(Linear(in_features=self.backbone_features, out_features=num_classes))

        if mode == "downstream":
            self.backbone = self.backbone.load_state_dict(torch.load(weights))

        self.init_weight()

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

    def init_weight(self):
        for name, m in self.backbone.named_modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                m.eps = 1e-5
                m.momentum = 0.1
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
