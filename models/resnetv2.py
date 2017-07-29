"""
ResNet with PyTorch

ResNet with pre-activation architectures

Implemented the following paper:
Kaiming He, et al. "Identity Mappings in Deep Residual Networks."
"""
import torch.nn as nn
import torch.nn.functional as F

cfg = [20, 32, 44, 56, 110, 164, 1001]

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class PreActBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        x = F.relu(self.bn1(x))
        identity = x
        x = self.conv1(x)
        x = self.conv2(F.relu(self.bn2(x)))
        x += self.shortcut(identity)
        return x

class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels,
                               kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias = False)
            )

    def forward(self, x):
        x = F.relu(self.bn1(x))
        identity = x
        x = self.conv1(x)
        x = self.conv2(F.relu(self.bn2(x)))
        x = self.conv3(F.relu(self.bn3(x)))
        x += self.shortcut(identity)
        return x

class ResNetV2(nn.Module):
    def __init__(self, depth, num_classes=10):
        assert depth in cfg, 'Error: model depth invalid or undefined!'

        super(ResNetV2, self).__init__()
        block = PreActBottleneck if depth > 110 else PreActBlock
        multiplier = (depth - 2) // 9 if depth > 110 else (depth - 2) // 6
        filters = [16, 16, 32, 64]

        self.conv1 = conv3x3(3, filters[0])
        self.in_channels = filters[0]
        self.conv2_x = self._make_layers(block, multiplier, filters[1], 1)
        self.conv3_x = self._make_layers(block, multiplier, filters[2], 2)
        self.conv4_x = self._make_layers(block, multiplier, filters[3], 2)
        self.bn4 = nn.BatchNorm2d(self.in_channels)
        self.fc = nn.Linear(self.in_channels, num_classes)

    def _make_layers(self, block, num_block, out_channels, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = block.expansion * out_channels
        for i in range(num_block - 1):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = block.expansion * out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = F.relu(self.bn4(x))
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
