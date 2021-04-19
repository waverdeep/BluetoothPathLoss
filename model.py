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

    def forward(self, x):
        return self.linear_relu_stack(x)


if __name__ == '__main__':
    model = VanillaNetwork()
    print("Model structure: ", model, "\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")