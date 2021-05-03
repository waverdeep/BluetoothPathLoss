import torch
import torch.nn as nn


def set_activation(name):
    if name == 'ReLU':
        return nn.ReLU()
    elif name == 'PReLU':
        return nn.PReLU()
    elif name == 'LeakyReLU':
        return nn.LeakyReLU()
    elif name == 'ELU':
        return nn.ELU()
    elif name == 'Tanh':
        return nn.Tanh()