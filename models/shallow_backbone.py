from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, AdaptiveMaxPool2d, AvgPool2d


class ShallowBackbone(nn.Module):
    def __init__(self):
        super(ShallowBackbone, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.bn1 = BatchNorm2d(num_features=32)
        self.relu1 = ReLU(inplace=True)
        self.pool1 = AvgPool2d(kernel_size=3, stride=2, padding=1)
        # 32x56x56

        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(num_features=32)
        self.relu2 = ReLU(inplace=True)
        self.pool2 = AvgPool2d(kernel_size=3, stride=2, padding=1)
        # 32x28x28

        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = BatchNorm2d(num_features=64)
        self.relu3 = ReLU(inplace=True)
        self.pool3 = AvgPool2d(kernel_size=3, stride=2, padding=1)
        # 64x14x14

        self.conv4 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = BatchNorm2d(num_features=128)
        self.relu4 = ReLU(inplace=True)
        self.pool4 = AvgPool2d(kernel_size=3, stride=2, padding=1)
        # 128x7x7

        self.global_pool = AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.global_pool(x)

        return x
