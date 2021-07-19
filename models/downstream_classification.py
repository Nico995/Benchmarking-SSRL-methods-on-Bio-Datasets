import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from .resnet_backbone import get_backbone


class DownstreamClassification(nn.Module):
    def __init__(self, pretrained_model, method, num_classes, code_size=256, weights=None):
        super(DownstreamClassification, self).__init__()
        self.methods_with_head = ['jigsaw', 'autoencoder']
        # Use the backbone from a previous pretext training
        self.backbone = pretrained_model.backbone

        if method not in self.methods_with_head:
            # Remove the last fc layer
            # TODO: This needs to be done only for some pretext models not for everyone (i.e. autoencoder doesn't need it)
            self.backbone = nn.Sequential(
                # removing the last convolutional layer
                *list(self.backbone.children())[:-1]
            )

        # Freeze the backbone
        for m in self.backbone.parameters():
            m.requires_grad = False

        # Attach a linear classifier to be trained
        self.linear_head = Linear(in_features=code_size, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.linear_head(x)
        return x
