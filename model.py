import torch
from torch import nn
from torch.autograd import Variable


class VanillaNetwork(nn.Module):
    def __init__(self, input_size=8):
        super(VanillaNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.PReLU(),
            nn.Linear(64, 128),
            nn.Dropout(0.3),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.PReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class VanillaLSTMNetwork(nn.Module):
    def __init__(self, input_size=2):
        super(VanillaLSTMNetwork, self).__init__()
        self.hidden_size = 128
        self.num_layers = 2
        self.lstm01 = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              batch_first=True)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.Dropout(0.3),
            nn.PReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).cuda()

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).cuda()
        output, (h_out, _) = self.lstm01(x, (h_0, c_0))
        output, _ = self.lstm01(x, (h_0, c_0))
        return self.linear_relu_stack(output[:, -1, :])


if __name__ == '__main__':
    kind = 'RNN'
    if kind == 'ANN':
        model = VanillaNetwork()
    elif kind == 'RNN':
        model = VanillaLSTMNetwork()
    print("Model structure: ", model, "\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")