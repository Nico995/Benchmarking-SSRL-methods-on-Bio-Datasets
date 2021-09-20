from torch import nn
from torch.nn import ConvTranspose2d, ReLU, Linear, BatchNorm2d, \
    Tanh, Sigmoid, Upsample, Conv2d


class Generator(nn.Module):
    """
    This is the decoder part of the AutoEncoder. It takes in the code from the Encoder, and outputs an image
    """

    def __init__(self, latent_dim, channel_factor=16):
        super(Generator, self).__init__()
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
        # self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        # print('Generator')
        # print('in ', x.shape)
        x = self.conv1(x)
        # print('c1 ', x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        # print('c2 ', x.shape)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        # print('c3 ', x.shape)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        # print('c4 ', x.shape)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        # print('c5 ', x.shape)
        x = self.activ(x)
        return x


class GeneratorUpsample(nn.Module):
    def __init__(self, latent_dim, channel_factor=16):
        super(GeneratorUpsample, self).__init__()

        # input 4x4

        self.up1 = Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = Conv2d(latent_dim, channel_factor*16, 3, 1, 2)
        # map 8x8

        self.up2 = Upsample(scale_factor=2, mode='bilinear')
        self.conv2 = Conv2d(channel_factor*16, channel_factor*8, 3, 1, 2)
        # map 16x16

        self.up3 = Upsample(scale_factor=2, mode='bilinear')
        self.conv3 = Conv2d(channel_factor*8, channel_factor*4, 3, 1, 2)
        # map 32x32

        self.up4 = Upsample(scale_factor=2, mode='bilinear')
        self.conv4 = Conv2d(channel_factor*4, channel_factor*2, 3, 1, 2)
        # map 64x64

        self.up5 = Upsample(scale_factor=2, mode='bilinear')
        self.conv5 = Conv2d(channel_factor*2, channel_factor, 3, 1, 2)
        # map 128x128

        self.up6 = Upsample(scale_factor=2, mode='bilinear')
        self.conv6 = Conv2d(channel_factor, 3, 3, 1, 2)
        # map 256x256


class Discriminator(nn.Module):
    def __init__(self, latent_dim, channel_factor=16):
        super(Discriminator, self).__init__()

        # input is (nc) x 224 x 224
        self.conv1 = Conv2d(3, channel_factor, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_factor)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf) x 112 x 112
        self.conv2 = Conv2d(channel_factor, channel_factor * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel_factor * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*2) x 56 x 56
        self.conv3 = Conv2d(channel_factor * 2, channel_factor * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel_factor * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*4) x 28 x 28
        self.conv4 = Conv2d(channel_factor * 4, channel_factor * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(channel_factor * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*8) x 14 x 14
        self.conv5 = Conv2d(channel_factor * 8, 1, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(1)
        # state size. 1 x 7 x 7
        self.fc = Linear(7 * 7, 2)
        # state size. 2

    def forward(self, x):
        # print('Discriminator')
        # print('in ', x.shape)
        x = self.conv1(x)
        # print('c1 ', x.shape)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        # print('c2 ', x.shape)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        # print('c3 ', x.shape)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        # print('c4 ', x.shape)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        # print('c5 ', x.shape)
        x = self.bn5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        # print('out ', x.shape)

        return x


class GAN(nn.Module):
    def __init__(self, num_classes, version='18', weights=None, mode="pretext"):
        super(GAN, self).__init__()

        self.backbone = Generator(256)

        if mode == "pretext":
            self.discriminator = Discriminator(256)
            # self.discriminator, out_features = get_backbone(version=version)
            # self.discriminator = Sequential(self.discriminator, Flatten(-3, -1), Linear(out_features, 2))
        else:
            # For downstream task, use a standard classifier and load pretrained weights
            NotImplemented()
