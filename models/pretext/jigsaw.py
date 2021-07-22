import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU, AdaptiveAvgPool1d

from ..resnet_backbone import get_backbone


class JigsawBackbone(nn.Module):

    def __init__(self, backbone):
        super(JigsawBackbone, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        """
        Args:
            x (torch.tenor): Batch tensor of expected size T, B, C, W, H, where
                T: number of Tiles (tipically = 9)
                B: Batch size
                C: number of Channels
                W, H: Width and Height
        """
        num_tiles, num_samples, c, h, w = x.size()
        features = []
        # Iterate over the number of tiles
        for t in range(num_tiles):
            # Extract features for each tile, for each image
            z = self.backbone(x[t])
            features.append(z)

        return z


class Jigsaw(nn.Module):
    """
    Performs SSRL predicting Jigsaw permutation.
    The jigsaw task consists in predicting which of the 100 pseudo random permutations has been applied to a specific image.
    The backbone is used as a feature extractor, with some pre-defined latent space dimension.
    Consequently, the resulting code is run through a classifier that predicts which one of the 100 permutation has been
    applied to a given image
    """

    def __init__(self, num_classes, version='18', weights=None, mode="pretext"):
        super(Jigsaw, self).__init__()

        self.backbone, self.backbone_features = get_backbone(version=version, weights=weights)
        self.backbone = JigsawBackbone(self.backbone)

        if mode == "pretext":
            # The factor of 9 is because in the forward method we concatenate
            # the 9 tiles b4 sending them to the classifier
            self.classifier = Sequential(
                Linear(9 * self.backbone_features, 1024),
                ReLU(inplace=True),
                Linear(1024, num_classes))
        else:
            # For downstream task, use a standard classifier and load pretrained weights
            self.backbone = self.backbone.load_state_dict(torch.load(weights))
            self.classifier = Sequential(
                Linear(in_features=self.backbone_features, out_features=num_classes))

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)
