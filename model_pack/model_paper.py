import torch
from torch import nn
from torch.autograd import Variable
from model_pack import model_config
import collections


class PastCRNN(nn.Module):
    def __init__(self, input_size=11,):
        super(PastCRNN, self).__init__()
        self.convolution_layer = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv01', nn.Conv1d(input_size, input_size, 3)),
                ]
            )
        )
        self.recurrent_layer = nn.Sequential(
            collections.OrderedDict(
                [
                    'lstm01', nn.LSTM(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True,
                                      bidirectional=False)
                ]
            )
        )

        self.dense = nn.Sequential(
            collections.OrderedDict(
                [
                    ('regression_linear01', nn.Linear(256, 64)),
                    ('dropout_01', nn.Dropout(0.3)),
                    ('regression_linear02', nn.Linear(64, 1))
                ]
            )
        )

    def forward(self, x):
        out = self.convolution_layer(x)
        out = out.transpose(1, 2)
        out, (h_out, _) = self.recurrent_layer(out) #, model_config.set_state_h_c(num_layer=2, size=5, hidden_size=128, cuda=False, cuda_num="cuda:0"))
        # print(out.size())
        return self.dense(out[:, -1, :])


class DilatedCRNNSmallV5(nn.Module):
    def __init__(self):
        super(DilatedCRNNSmallV5, self).__init__()
        self.encoder = nn.Sequential(
            collections.OrderedDict(
                [
                    ('encoder_conv01', nn.Conv1d(3, 128, kernel_size=3, stride=1, padding=1, dilation=2)),
                    ('encoder_bn01', nn.BatchNorm1d(128)),
                    ('encoder_relu01', nn.PReLU()),
                    ('encoder_conv02', nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, dilation=2)),
                    ('encoder_bn02', nn.BatchNorm1d(256)),
                    ('encoder_relu02', nn.PReLU()),
                ]
            )
        )

        self.regression = nn.Sequential(
            collections.OrderedDict(
                [
                    ('regression_lstm01', nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True,
                                                  bidirectional=False)),
                ]
            )
        )

        self.post = nn.Sequential(
            collections.OrderedDict(
                [
                    ('regression_linear01', nn.Linear(128, 64)),
                    ('regression_bn01', nn.BatchNorm1d(64)),
                    ('regression_linear02', nn.Linear(64, 1))
                ]
            )
        )

    def forward(self, x):
        out = self.encoder(x)
        out = out.transpose(1, 2)
        out, (h_out, _) = self.regression(out) #, model_config.set_state_h_c(num_layer=2, size=5, hidden_size=128, cuda=False, cuda_num="cuda:0"))
        # print(out.size())
        return self.post(out[:, -1, :])
