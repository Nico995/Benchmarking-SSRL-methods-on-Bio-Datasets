import torch
from torch import nn
from torch.nn import Linear, Sequential, Conv2d, MaxPool2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from ..resnet_backbone import get_backbone
from ..shallow_backbone import ShallowBackbone


class Supervised(nn.Module):
    """
    Classic fully supervised training.
    """

    def __init__(self, num_classes, version='18', weights=None, mode="pretext"):
        super(Supervised, self).__init__()

        self.backbone, self.backbone_features = get_backbone(version=version, weights=weights)
        # self.backbone = ShallowBackbone()
        # self.backbone_features = 128
        self.classifier = Linear(in_features=self.backbone_features, out_features=num_classes)

        if mode == "downstream":
            self.backbone = self.backbone.load_state_dict(torch.load(weights))

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)

