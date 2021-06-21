import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from .resnet_backbone import get_backbone


class DownstreamClassification(nn.Module):
    def __init__(self, pretrained_model, num_classes, code_size=512, weights=None):
        super(DownstreamClassification, self).__init__()
        # Use the backbone from a previous pretext training
        self.backbone = pretrained_model.backbone
        # Remove the last fc layer
        self.backbone = nn.Sequential(
            # removing the last convolutional layer
            *list(self.backbone.children())[:-1]
        )
        # Freeze the backbone
        for m in self.backbone.parameters():
            m.requires_grad = False

        ##############
        # from torchsummary import summary
        # print(summary(self.backbone.cuda(), (3, 64, 64)))
        # exit()
        ##############
        # Attach a linear classifier to be trained
        self.linear_head = Linear(in_features=code_size, out_features=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.linear_head(x)
        return x
