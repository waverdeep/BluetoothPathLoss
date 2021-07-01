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


def set_recurrent_layer(name, input_size, hidden_size, num_layers, batch_first=True, bidirectional=False):
    return nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first,
                   bidirectional=bidirectional)