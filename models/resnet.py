"""
ResNet with PyTorch

ResNet Basic && ResNet with bottelneck architectures [1]
ResNet with pre-activation architectures [2]

Implemented the following papers:
[1] Kaiming He, et al. "Deep Residual Learning for Image Recognition".
[2] Kaiming He, et al. "Identity Mappings in Deep Residual Networks".
"""
import torch.nn as nn
import torch.nn.functional as F

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
                               kernel_size=1, bias=False)
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
                          kernel_size=1, stride=stride, bias = False),
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

class PreActBlock(nn.Module):
    """ Pre-activation version of BasicBlock """
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
    """ Pre-activation version of Bottleneck """
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

class ResNet(nn.Module):
    def __init__(self, block, multiplier, filters, pre_act_block=True, num_classes=10):
        super(ResNet, self).__init__()
        if pre_act_block:
            block = self._get_pre_act_block(block)
        
        self.conv1 = conv3x3(3, filters[0])
        self.in_channels = filters[0]
        self.conv2_x = self._make_layers(block, multiplier, filters[1],
                                         stride=2, pre_act=not pre_act_block)
        self.conv3_x = self._make_layers(block, multiplier, filters[2],
                                         stride=2)
        self.conv4_x = self._make_layers(block, multiplier, filters[3],
                                         stride=2, post_act=pre_act_block)
        self.fc = nn.Linear(self.in_channels, num_classes)

    def _get_pre_act_block(self, block):
        block = PreActBlock if block == BasicBlock else PreActBottleneck
        return block

    def _make_layers(self, block, num_block, out_channels, stride,
                     pre_act=False, post_act=False):
        layers = []

        if pre_act:
            layers.append(nn.BatchNorm2d(self.in_channels))
            layers.append(nn.ReLU())

        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = block.expansion * out_channels
        for i in range(num_block - 1):
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = block.expansion * out_channels

        if post_act:
            layers.append(nn.BatchNorm2d(self.in_channels))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
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
    return ResNet(BasicBlock, 5, [16, 16, 32, 64])

def ResNet44():
    return ResNet(BasicBlock, 7, [16, 16, 32, 64])

def ResNet56():
    return ResNet(Bottleneck, 9, [16, 16, 32, 64])

def ResNet110():
    return ResNet(BasicBlock, 18, [16, 16, 32, 64])

def ResNet164():
    return ResNet(Bottleneck, 18, [16, 16, 32, 64])

def ResNet1001():
    return ResNet(Bottleneck, 111, [16, 16, 32, 64])
