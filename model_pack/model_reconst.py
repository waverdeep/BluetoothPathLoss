import torch
from torch import nn
from torch.autograd import Variable
from model_pack import model_config
import collections


class DilatedCRNN(nn.Module):
    def __init__(self):
        super(DilatedCRNN, self).__init__()
        self.encoder = nn.Sequential(
            collections.OrderedDict(
                [
                    ('encoder_conv01', nn.Conv1d(11, 48, kernel_size=3, stride=1, dilation=2, padding=1)),
                    ('encoder_bn01', nn.BatchNorm1d(48)),
                    ('encoder_relu01', nn.ReLU()),
                    ('encoder_conv02', nn.Conv1d(48, 48, kernel_size=5, stride=1, dilation=2, padding=1)),
                    ('encoder_bn02', nn.BatchNorm1d(48)),
                    ('encoder_relu02', nn.ReLU()),
                    ('encoder_conv03', nn.Conv1d(48, 48, kernel_size=3, stride=1, dilation=2, padding=1)),
                    ('encoder_bn03', nn.BatchNorm1d(48)),
                    ('encoder_relu03', nn.ReLU()),
                ]
            )
        )

        self.regression = nn.Sequential(
            collections.OrderedDict(
                [
                    ('regression_lstm01', nn.LSTM(input_size=48, hidden_size=128, num_layers=2, batch_first=True,
                                                  bidirectional=True)),
                ]
            )
        )

        self.post = nn.Sequential(
            collections.OrderedDict(
                [
                    ('regression_linear01', nn.Linear(256, 128)),
                    ('regression_bn01', nn.BatchNorm1d(128)),
                    ('regression_linear02', nn.Linear(128, 1))
                ]
            )
        )

    def forward(self, x):
        out = self.encoder(x)
        out = out.transpose(1, 2)
        out, (h_out, _) = self.regression(out) #, model_config.set_state_h_c(num_layer=2, size=5, hidden_size=128, cuda=False, cuda_num="cuda:0"))
        return self.post(out[:, -1, :])


class CRNN(nn.Module):
    def __init__(self, input_size=11, model_type='LSTM', activation='ReLU', bidirectional=False,
                 hidden_size=64, num_layers=1, linear_layers=None, dropout_rate=0.5, use_cuda=True,
                 convolution_layer=3, use_batch_norm=True, cuda_num='cuda:0'):
        super(CRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.linear_layers = linear_layers
        self.linear_layers.insert(0, hidden_size)
        self.use_cuda = use_cuda
        self.cuda_num = cuda_num
        self.conv1d_stack = nn.Sequential()
        self.conv1d_stack = model_config.build_conv1d_layer(self.conv1d_stack, convolution_layers=convolution_layer,
                                                            input_size=input_size)
        self.rnn_layer01 = model_config.set_recurrent_layer(name=model_type, input_size=input_size,
                                                            hidden_size=hidden_size, num_layers=num_layers,
                                                            batch_first=True, bidirectional=bidirectional)
        self.linear_stack = nn.Sequential()
        self.linear_stack = model_config.build_linear_layer(layer=self.linear_stack, linear_layers=self.linear_layers,
                                                            activation=activation, dropout_rate=dropout_rate,
                                                            use_batch_norm=use_batch_norm)

    def forward(self, x):
        out = self.conv1d_stack(x)
        out = out.transpose(1, 2)
        out, (h_out, _) = self.rnn_layer01(out, model_config.set_state_h_c(num_layer=self.num_layers, size=x.size(0),
                                                                           hidden_size=self.hidden_size,
                                                                           cuda=self.use_cuda, cuda_num=self.cuda_num))
        return self.linear_stack(out[:, -1, :])


def model_load(model_configure):
    if model_configure['model_type'] == 'Custom_CRNN':
        nn_model = CRNN(
            input_size=model_configure['input_size'],
            model_type=model_configure['model_type'],
            activation=model_configure['activation'],
            bidirectional=model_configure['bidirectional'],
            hidden_size=model_configure['hidden_size'],
            num_layers=model_configure['num_layers'],
            linear_layers=model_configure['linear_layers'],
            dropout_rate=model_configure['dropout_rate'],
            use_cuda=model_configure['use_cuda'],
            convolution_layer=model_configure['convolution_layer'],
            cuda_num=model_configure['cuda_num'],
            use_batch_norm=True if 'use_batch_norm' in model_configure else False # 일단 해당 속성이 있으면 하는거로
        )
    elif model_configure['model_type'] == 'DilatedCRNN':
        nn_model = DilatedCRNN()

    if model_configure['use_cuda']:
        device = torch.device(model_configure['cuda_num'])
        nn_model = nn_model.to(device)

    if 'checkpoint_path' in model_configure:
        if model_configure['use_cuda']:
            checkpoint = torch.load(model_configure['checkpoint_path'])
            nn_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            ### gpu로 생성된 checkpoint를 cpu로 변경해서 사용하기 위한 코드 작성
            device = torch.device('cpu')
            checkpoint = torch.load(model_configure['checkpoint_path'], map_location=device)
            nn_model.load_state_dict(checkpoint['model_state_dict'])

    print("Model structure: ", nn_model, "\n\n")
    return nn_model


if __name__ == '__main__':
    model = DilatedCRNN()
    data = torch.randn(10, 11, 15)
    out_data = model(data)
    print(out_data.size())
