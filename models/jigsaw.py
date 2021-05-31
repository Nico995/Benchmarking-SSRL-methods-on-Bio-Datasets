import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from .resnet_backbone import get_backbone

'''
The jigsaw task consists in predicting which of the 100 pseudo random permutations has been applied to a specific image.
The backbone is used as a feature extractor, with some pre-defined latent space dimension. 
Consequently, the resulting code is run through a classifier that predicts which one of the 100 permutation has been 
applied to a given image (=> classifier output shape: [B, #_perm] )
'''


class Jigsaw(nn.Module):

    def __init__(self, code_size=256, num_perms=100, version='18'):
        super(Jigsaw, self).__init__()
        # Backbone of jigsaw model
        self.backbone = get_backbone(out_features=code_size, version=version)

        # Classifier for jigsaw task
        self.classifier = Sequential(
            Linear(9 * code_size, 1024),
            ReLU(inplace=True),
            Linear(1024, num_perms))

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
        x = torch.cat(features, 1)
        x = self.classifier(x)

        return x
