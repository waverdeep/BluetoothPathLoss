import torch
from torch import nn


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
        self.lstm01 = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=4, batch_first=True)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.3),
            nn.PReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        output, _ = self.lstm01(x)
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