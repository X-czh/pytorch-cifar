"""
ResNet with PyTorch

ResNet basic and bottelneck architectures

Implemented the following papers:
Kaiming He, et al. "Deep Residual Learning for Image Recognition."
"""
import math

import torch.nn as nn
import torch.nn.functional as F

cfg = [20, 32, 44, 56, 110, 164, 1001]

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(identity)
        x = F.relu(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x): 
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += self.shortcut(identity)
        x = F.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, depth, num_classes=10):
        assert depth in cfg, 'Error: model depth invalid or undefined!'

        super(ResNet, self).__init__()
        block = Bottleneck if depth > 110 else BasicBlock
        multiplier = (depth - 2) // 9 if depth > 110 else (depth - 2) // 6
        filters = [16, 16, 32, 64]

        self.conv1 = conv3x3(3, filters[0])
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.in_channels = filters[0]
        self.conv2_x = self._make_layers(block, multiplier, filters[1], 1)
        self.conv3_x = self._make_layers(block, multiplier, filters[2], 2)
        self.conv4_x = self._make_layers(block, multiplier, filters[3], 2)
        self.fc = nn.Linear(self.in_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layers(self, block, num_block, out_channels, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = block.expansion * out_channels
        for i in range(num_block - 1):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = block.expansion * out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
