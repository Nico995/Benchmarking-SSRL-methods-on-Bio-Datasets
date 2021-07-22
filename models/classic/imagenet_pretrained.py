import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from models.resnet_backbone import get_backbone


class ImagenetPretrained(nn.Module):

    def __init__(self, num_classes, version='18', weights=None):
        super(ImagenetPretrained, self).__init__()
        self.backbone, self.backbone_features = get_backbone(version=version, weights=weights, pretrained=True)
        self.classifier = Sequential(Linear(in_features=self.backbone_features, out_features=num_classes))

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)
