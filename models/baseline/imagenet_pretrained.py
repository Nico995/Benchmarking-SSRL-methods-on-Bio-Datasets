import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from models.resnet_backbone import get_backbone


class ImagenetPretrained(nn.Module):

    def __init__(self, num_classes, version='18', weights=None, mode="pretext"):
        super(ImagenetPretrained, self).__init__()
        self.backbone, self.backbone_features = get_backbone(version=version, weights=weights, pretrained=True)
        self.classifier = Linear(in_features=self.backbone_features, out_features=num_classes)

        if mode == "downstream":
            self.backbone = self.backbone.load_state_dict(torch.load(weights))

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)
