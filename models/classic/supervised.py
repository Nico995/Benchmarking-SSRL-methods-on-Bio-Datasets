from torch import nn
from torch.nn import Linear
from ..resnet_backbone import get_backbone


class Supervised(nn.Module):
    """
    Classic fully supervised training.
    """

    def __init__(self, num_classes, version='18', weights=None):
        super(Supervised, self).__init__()
        self.backbone, self.backbone_features = get_backbone(version=version, weights=weights)
        self.classifier = Linear(in_features=self.backbone_features, out_features=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

