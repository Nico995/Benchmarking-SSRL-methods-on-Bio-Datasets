from torch import nn
from torch.nn import Sequential, ConvTranspose2d, BatchNorm2d, ReLU, Tanh

from models.resnet_backbone import get_backbone


class Decoder(nn.Module):
    """
    This is the decoder part of the AutoEncoder. It takes in the code from the Encoder, and outputs an image
    """

    def __init__(self, latent_dim, channel_factor=64):
        super(Decoder, self).__init__()
        self.channels = 3
        self.nn = Sequential(

            ConvTranspose2d(latent_dim, channel_factor * 8, 4, 1, 0, bias=False),
            BatchNorm2d(channel_factor * 8),
            ReLU(True),
            # out_shape: channel_factor * 8, 4, 4
            ConvTranspose2d(channel_factor * 8, channel_factor * 4, 4, 2, 1, bias=False),
            BatchNorm2d(channel_factor * 4),
            ReLU(True),
            # out_shape: channel_factor * 4, 8, 8
            ConvTranspose2d(channel_factor * 4, channel_factor * 2, 4, 2, 1, bias=False),
            BatchNorm2d(channel_factor * 2),
            ReLU(True),
            # out_shape: channel_factor * 2, 16, 16
            ConvTranspose2d(channel_factor * 2, channel_factor, 4, 2, 1, bias=False),
            BatchNorm2d(channel_factor),
            ReLU(True),
            # out_shape: channel_factor, 32, 32
            ConvTranspose2d(channel_factor, self.channels, 4, 2, 1, bias=False),
            Tanh()
            # out_shape: channels, 64, 64
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        return self.nn(x)


class AutoEncoder(nn.Module):
    def __init__(self, num_classes, latent_dim=256, channel_factor=64, version='18', weights=None):
        super(AutoEncoder, self).__init__()

        self.backbone = get_backbone(out_features=latent_dim, version=version)
        self.decoder = Decoder(latent_dim=latent_dim, channel_factor=channel_factor)

    def forward(self, x):
        code = self.backbone(x)
        return self.decoder(code)
