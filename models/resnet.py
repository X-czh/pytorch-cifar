"""
ResNet with PyTorch
"""
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBLock, self).__init__()
        self.conv1 = nn.Conv2d()

class ResNet():
    return x