import torch
from torch import nn
from torch.autograd import Variable
import model_config


class VanillaNetworkCustom(nn.Module):
    def __init__(self, input_size=8, activation='PReLU', layer=[64, 128, 64, 1]):
        super(VanillaNetworkCustom, self).__init__()
        self.activation = activation
        layer.insert(0, input_size)
        modules = []
        for grip in range(len(layer)-1):
            modules.append(nn.Linear(layer[grip], layer[grip+1]))

            if (grip + 1) != len(layer):
                modules.append(nn.Dropout(0.3))
                modules.append(model_config.set_activation(self.activation))

        self.linear_relu_stack = nn.Sequential(*modules)

    def forward(self, x):
        return self.linear_relu_stack(x)


class VanillaNetwork(nn.Module):
    def __init__(self, input_size=8, activation='PReLU'):
        super(VanillaNetwork, self).__init__()
        self.activation = activation
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            model_config.set_activation(self.activation),
            nn.Linear(64, 64),
            nn.Dropout(0.3),
            model_config.set_activation(self.activation),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class VanillaRecurrentNetwork(nn.Module):
    def __init__(self, input_size=8, recurrent_model='LSTM', activation='PReLU', bidirectional=False):
        super(VanillaRecurrentNetwork, self).__init__()
        self.activation = activation
        self.hidden_size = 64
        self.num_layers = 1
        self.lstm01 = model_config.set_recurrent_layer(name=recurrent_model, input_size=input_size,
                                                       hidden_size=self.hidden_size, num_layers=self.num_layers,
                                                       batch_first=True, bidirectional=bidirectional)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.Dropout(0.3),
            model_config.set_activation(self.activation),
            nn.Linear(64, 32),
            nn.Dropout(0.3),
            model_config.set_activation(self.activation),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).cuda()
        output, (h_out, _) = self.lstm01(x, (h_0, c_0))
        output, _ = self.lstm01(x, (h_0, c_0))
        return self.linear_relu_stack(output[:, -1, :])


class VanillaCNNRNNNetwork(nn.Module):
    def __init__(self, input_size=8, recurrent_model='LSTM', activation='PReLU', bidirectional=False):
        super(VanillaCNNRNNNetwork, self).__init__()
        # convolution
        self.num_layers = 1
        self.hidden_size = 256
        self.conv1d_layer = nn.Conv1d(8, 8, 3)
        self.activation = activation
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                  batch_first=True)
        self.linear_layer1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.3)
        self.activation = model_config.set_activation(self.activation)
        self.linear_layer2 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.conv1d_layer(x)
        print(out.shape)
        out = out.transpose(1, 2)
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).cuda()
        out, (h_out, _) = self.lstm_layer(out, (h_0, c_0))
        out = self.activation(self.dropout(self.linear_layer1(out)))
        out = self.linear_layer2(out)
        return out


if __name__ == '__main__':
    kind = 'CNNRNN'
    if kind == 'ANN':
        model = VanillaNetwork()
    elif kind == 'RNN':
        model = VanillaRecurrentNetwork()
    elif kind == 'CNNRNN':
        model = VanillaCNNRNNNetwork()
    print("Model structure: ", model, "\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    # model test
    x = torch.empty(62, 8, 15)
    model(x)