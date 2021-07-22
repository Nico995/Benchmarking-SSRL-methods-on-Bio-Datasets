from torch import nn
from torch.nn import ConvTranspose2d, ReLU, Sigmoid, Conv2d


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
    def __init__(self, num_classes, version='18', weights=None):
        super(AutoEncoder, self).__init__()

        self.backbone = SmallEncoder()
        self.decoder = SmallDecoder()

    def forward(self, x):
        code = self.backbone(x)
        return self.decoder(code)
