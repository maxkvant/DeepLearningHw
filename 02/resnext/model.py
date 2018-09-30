import torch.nn as nn

__all__ = ['ResNeXt', 'resnext50_32']

class ResNeXtBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, cardinality=1, downsample=None):
        super(ResNeXtBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1) # Is "bias = False" needed?
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

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

class ResNeXt(nn.Module):
    def __init__(self, in_channels, num_classes, layers=(3, 4, 6, 3), block=ResNeXtBlock, cardinality=32):
        super(ResNeXt, self).__init__()

        self.last_channels = 64
        self.cardinality = cardinality

        self.conv1 = nn.Conv2d(in_channels, self.last_channels, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(self.last_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], cardinality=cardinality)
        self.layer2 = self._make_layer(block, 128, layers[1], cardinality=cardinality, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cardinality=cardinality, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cardinality=cardinality, stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(self.last_channels * block.expansion, num_classes)

    def _make_layer(self, block, channels, blocks, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.last_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.last_channels, channels * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channels * block.expansion)
            )

        layers =  [block(self.last_channels, channels, stride=stride, cardinality=cardinality, downsample=downsample)]
        self.last_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.last_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext50_32(in_channels, num_classes, pretrained_weights=None):
    model = ResNeXt(in_channels, num_classes=num_classes)
    if pretrained_weights is not None:
        model.load_state_dict(pretrained_weights)
    return model