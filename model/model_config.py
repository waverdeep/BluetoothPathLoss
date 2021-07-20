import torch
from torch.autograd import Variable
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
    else:
        return nn.ReLU()


def n_gram(data_list, n=2):
    result = []
    for i in range(len(data_list) - (n - 1)):
        result.append(data_list[i:i+n])
    return result


# param type : [11, 32, 64, 32, 1]
def build_linear_layer(layer, linear_layers, activation, dropout_rate):
    grapped_linear = n_gram(linear_layers)
    for idx, line in enumerate(grapped_linear):
        layer.add_module("linear_{}".format(idx), nn.Linear(line[0], line[1]))
        if line[1] != 1:
            layer.add_module("activation_{}".format(idx), set_activation(activation))
            layer.add_module("dropout_{}".format(idx), nn.Dropout(dropout_rate))
    return layer


def build_conv1d_layer(layer, convolution_layers, input_size):
    for i in range(convolution_layers):
        layer.add_module("conv1d_{}".format(i), nn.Conv1d(input_size, input_size, 3))

    return layer


def set_state_h_c(num_layer, hidden_size, size, cuda):
    h_0 = Variable(torch.zeros(num_layer, size, hidden_size))
    c_0 = Variable(torch.zeros(num_layer, size, hidden_size))
    if cuda:
        h_0 = h_0.cuda()
        c_0 = c_0.cuda()
    return (h_0, c_0)


def set_recurrent_layer(name, input_size, hidden_size, num_layers, batch_first=True, bidirectional=False):
    return nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first,
                   bidirectional=bidirectional)