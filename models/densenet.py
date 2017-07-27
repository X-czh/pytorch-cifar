"""
DenseNet with PyTorch

DenseNet Basic && DenseNet-BC (with Bottleneck and Compression) architectures

Implemented the following paper:
Gao Huang, et al. "Densely Connected Convolutional Networks".
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        in_data = x
        x = self.conv(F.relu(self.bn(x)))
        x = torch.cat([x, in_data], 1)
        return x 

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        in_data = x
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        x = torch.cat([x, in_data], 1)
        return x

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = F.avg_pool2d(x, 2)
        return x

class DenseNet(nn.Module):
    def __init__(self, depth, growth_rate, bottleneck=True, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        
        num_block = (depth - 4) // 3
        if bottleneck:
            num_block //= 2

        in_channels = 3
        out_channels = 2*growth_rate
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        in_channels = out_channels

        self.dense1 = self._make_layers(in_channels, growth_rate, num_block, bottleneck)
        in_channels += num_block * growth_rate
        out_channels = int(math.floor(reduction * in_channels))
        self.trans1 = Transition(in_channels, out_channels)
        in_channels = out_channels
        
        self.dense2 = self._make_layers(in_channels, growth_rate, num_block, bottleneck)
        in_channels += num_block * growth_rate
        out_channels = int(math.floor(reduction * in_channels))
        self.trans2 = Transition(in_channels, out_channels)
        in_channels = out_channels

        self.dense3 = self._make_layers(in_channels, growth_rate, num_block, bottleneck)
        in_channels += num_block * growth_rate

        self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, num_classes)

    def _make_layers(self, in_channels, growth_rate, num_block, bottelneck):
        layers = []
        block = Bottleneck if bottelneck else BasicBlock
        for i in range(num_block):
            layers.append(block(in_channels, growth_rate))
            in_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = F.avg_pool2d(F.relu(self.bn(x)), 8)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def DenseNet40():
    return DenseNet(40, 12, bottleneck=False)

def DenseNet100():
    return DenseNet(100, 12)

def DenseNet250():
    return DenseNet(250, 24)

def DenseNet190():
    return DenseNet(190, 48)

def DenseNet10():
    """ Low memory consumption, just for testing """
    return DenseNet(10, 12)
