import torch
from torch import nn


class VanillaNetwork(nn.Module):
    def __init__(self):
        super(VanillaNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
