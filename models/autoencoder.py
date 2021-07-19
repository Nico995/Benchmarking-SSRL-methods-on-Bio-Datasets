import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import ConvTranspose2d, BatchNorm1d, BatchNorm2d, ReLU, Tanh, Linear, Sequential, Sigmoid, Conv2d

from models.resnet_backbone import get_backbone


class Decoder(nn.Module):
    """
    This is the decoder part of the AutoEncoder. It takes in the code from the Encoder, and outputs an image
    """

    def __init__(self, latent_dim, channel_factor=64):
        super(Decoder, self).__init__()
        self.channels = 3

        self.conv1 = ConvTranspose2d(latent_dim, channel_factor * 8, 4, 2, 1, bias=False)
        self.bn1 = BatchNorm2d(channel_factor * 8)
        self.relu1 = ReLU(True)
        # out_shape: channel_factor * 8, 4, 4
        self.conv2 = ConvTranspose2d(channel_factor * 8, channel_factor * 4, 4, 2, 1, bias=False)
        self.bn2 = BatchNorm2d(channel_factor * 4)
        self.relu2 = ReLU(True)
        # out_shape: channel_factor * 4, 8, 8
        self.conv3 = ConvTranspose2d(channel_factor * 4, channel_factor * 2, 4, 2, 1, bias=False)
        self.bn3 = BatchNorm2d(channel_factor * 2)
        self.relu3 = ReLU(True)
        # out_shape: channel_factor * 2, 16, 16
        self.conv4 = ConvTranspose2d(channel_factor * 2, channel_factor, 4, 2, 1, bias=False)
        self.bn4 = BatchNorm2d(channel_factor)
        self.relu4 = ReLU(True)
        # out_shape: channel_factor, 32, 32
        self.conv5 = ConvTranspose2d(channel_factor, self.channels, 4, 2, 1, bias=False)
        self.activ = Sigmoid()
        # out_shape: channels, 64, 64

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), 10, 7, 7)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.activ(x)
        return x


class SmallEncoder(nn.Module):
    def __init__(self):
        super(SmallEncoder, self).__init__()

        # 3 x 224 x 224
        self.conv1 = Conv2d(3, 12, 4, stride=2, padding=1)
        # 12 x 112 x 112
        self.relu1 = ReLU()
        self.conv2 = Conv2d(12, 24, 4, stride=2, padding=1)
        # 24 x 56 x 56
        self.relu2 = ReLU()
        self.conv3 = Conv2d(24, 48, 4, stride=2, padding=1)
        # 48 x 28 x 28
        self.relu3 = ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.shape[0], -1)
        return x


class SmallDecoder(nn.Module):
    def __init__(self):
        super(SmallDecoder, self).__init__()

        self.convt1 = ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        self.relu1 = ReLU()
        self.convt2 = ConvTranspose2d(24, 12, 4, stride=2, padding=1)
        self.relu2 = ReLU()
        self.convt3 = ConvTranspose2d(12, 3, 4, stride=2, padding=1)
        self.sigm = Sigmoid()

    def forward(self, x):
        x = x.view(x.shape[0], 48, 28, 28)
        x = self.convt1(x)
        x = self.relu1(x)
        x = self.convt2(x)
        x = self.relu2(x)
        x = self.convt3(x)
        x = self.sigm(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self, num_classes, latent_dim=10, out_features=490, channel_factor=64, version='18', weights=None):
        super(AutoEncoder, self).__init__()

        self.backbone = SmallEncoder()
        self.decoder = SmallDecoder()

        if weights:
            self.load_state_dict(torch.load(weights))
            self.backbone = nn.Sequential(
                self.backbone,
                Linear(37632, 1024),
                Linear(1024, 512),
            )
        # print(self.backbone)

        # self.backbone = get_backbone(version=version, out_features=out_features)
        # self.sigmoid = Sigmoid()
        # self.decoder = Decoder(latent_dim=latent_dim, channel_factor=64)

    def forward(self, x):
        code = self.backbone(x)
        # code = self.sigmoid(code)
        return self.decoder(code)
