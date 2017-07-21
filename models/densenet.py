"""
DenseNet with PyTorch
"""
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBLock, self).__init__()
        self.conv1 = nn.Conv2d()

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()


    def _make_layers(self, block):
        layers = []
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        return x
