import torch
from torch import nn
from torch.nn import ConvTranspose2d, ReLU, Sigmoid, Conv2d, Sequential, AdaptiveAvgPool2d, Linear, BatchNorm2d, Flatten
from ..resnet_backbone import get_backbone


class Decoder(nn.Module):

    """
    This is the decoder part of the AutoEncoder. It takes in the code from the Encoder, and outputs an image
    """
    def __init__(self, latent_dim, channel_factor=32):
        super(Decoder, self).__init__()
        self.channels = 3

        self.conv1 = ConvTranspose2d(latent_dim, channel_factor * 8, 4, 2, 1, bias=False)
        self.bn1 = BatchNorm2d(channel_factor * 8)
        self.relu1 = ReLU(True)
        # # out_shape: channel_factor * 8, 4, 4
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
        x = x.view(x.size(0), 512, 7, 7)
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
    def __init__(self, num_classes, version='18', weights=None, mode="pretext"):
        super(AutoEncoder, self).__init__()

        self.backbone, self.backbone_features = get_backbone(version)
        self.backbone = Sequential(*list(self.backbone.children())[:-1])

        # self.backbone = SmallEncoder()
        # self.backbone_features = 37632
        # The naming of classifier for the decoder network is confusing, but
        # needed to keep coherence throughout the classes. This could highly benefit
        # from some quality-python inheritance-centered refactoring
        if mode == "pretext":
            self.classifier = Decoder(latent_dim=self.backbone_features)
        else:
            # For downstream task, use a standard classifier and load pretrained weights
            self.backbone.load_state_dict(torch.load(weights))
            self.classifier = Sequential(
                AdaptiveAvgPool2d(output_size=(1, 1)),
                Flatten(-3, -1),
                Linear(in_features=self.backbone_features, out_features=num_classes))

    def forward(self, x):
        code = self.backbone(x)
        return self.classifier(code)
