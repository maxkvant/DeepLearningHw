import torch.nn as nn


class ResNeXtBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None, cardinality=32):
        super(ResNeXtBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=stride,
                               groups=cardinality)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.downsample = downsample
        self.cardinality = cardinality

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out