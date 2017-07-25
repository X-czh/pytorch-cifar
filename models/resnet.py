"""
ResNet with PyTorch
"""
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(BasicBlock, self).__init__()

        self.downsample = downsample

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += self.shortcut(residual)
        x = F.relu(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(Bottleneck, self).__init__()

        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)

        self.shortcut = nn.Sequential()
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias = False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += self.shortcut(residual)
        x = F.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, multiplier, filters):
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, filters[0])
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.in_channels = filters[0]
        self.conv2_x = self._make_layers(block, multiplier, filters[1])
        self.conv3_x = self._make_layers(block, multiplier, filters[2])
        self.conv4_x = self._make_layers(block, multiplier, filters[3])
        self.fc = nn.Linear(self.in_channels * block.expansion, 10)

    def _make_layers(self, block, num_block, out_channels):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=2, downsample=True))
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
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet20():
    return ResNet(BasicBlock, 3, [16, 16, 32, 64])

def ResNet32():
    return ResNet(BasicBlock, 4, [16, 16, 32, 64])

def ResNet44():
    return ResNet(BasicBlock, 5, [16, 16, 32, 64])

def ResNet56():
    return ResNet(Bottleneck, 6, [16, 16, 32, 64])

def ResNet110():
    return ResNet(Bottleneck, 18, [16, 16, 32, 64])
