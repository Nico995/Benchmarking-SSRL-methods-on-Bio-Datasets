import torch
from torch import nn
from torch.nn import Sequential, Linear, AdaptiveAvgPool1d

from models.resnet_backbone import get_backbone


class Rotation(nn.Module):
    """
    Performs SSRL predicting image rotation.
    Standard implementation rotates the image of 0, 90, 180, 270 degrees (4 classes)
    """

    def __init__(self, num_classes, version='18', weights=None, mode="pretext"):
        super(Rotation, self).__init__()

        self.backbone, self.backbone_features = get_backbone(version=version, weights=weights)

        if mode == "pretext":
            self.classifier = Sequential(Linear(in_features=self.backbone_features, out_features=num_classes))
        else:
            # For downstream task, use a standard classifier and load pretrained weights
            self.backbone = self.backbone.load_state_dict(torch.load(weights))
            self.classifier = Sequential(
                Sequential(Linear(in_features=self.backbone_features, out_features=num_classes)))

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)
