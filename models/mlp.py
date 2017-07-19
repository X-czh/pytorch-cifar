"""
a simple multilayer perceptron
"""
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = F.Linear(32*32*3, 128)
        self.fc2 = F.Linear(128, 64)
        self.fc3 = F.linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    