"""
ResNet with PyTorch
"""
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(BasicBlock, self).__init__()

        self.expansion = 1
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
                nn.Conv2d(in_channels,self.expansion*out_channels,
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
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(Bottleneck, self).__init__()

        self.expansion = 4
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
    def __init__(self, block, num_block):
        super(ResNet, self).__init__()

        self.in_channels = 16

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.conv2_x = self._make_layers(block, num_block, 16)
        self.conv3_x = self._make_layers(block, num_block, 32)
        self.conv4_x = self._make_layers(block, num_block, 64)
        self.fc = nn.Linear(512*block.expansion, 10)
    
    def _make_layers(self, block, num_block, out_channels):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=2, downsample=True))
        for i in range(num_block - 1):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = block.expansion * out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet20():
    return ResNet(BasicBlock, 3)