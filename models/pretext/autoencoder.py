import torch
from torch import nn
from torch.nn import ConvTranspose2d, ReLU, Sigmoid, Conv2d, Sequential, AdaptiveAvgPool1d, Linear


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

        self.backbone = SmallEncoder()

        # The naming of classifier for the decoder network is confusing, but
        # needed to keep coherence throughout the classes. This could highly benefit
        # from some quality-python inheritance-centered refactoring
        if mode == "pretext":
            self.classifier = SmallDecoder()
        else:
            # For downstream task, use a standard classifier and load pretrained weights
            self.backbone = self.backbone.load_state_dict(torch.load(weights))
            self.classifier = Sequential(
                AdaptiveAvgPool1d((1, 1)),
                Linear(in_features=self.backbone_features, out_features=num_classes))

    def forward(self, x):
        code = self.backbone(x)
        return self.classifier(code)
