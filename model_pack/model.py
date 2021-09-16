import torch
from torch import nn
from torch.autograd import Variable
import model_config


class Custom_DNN(nn.Module):
    def __init__(self, linear_layers=None, activation='ReLU', dropout_rate=0.5, cuda_num='cuda:0'):
        super(Custom_DNN, self).__init__()
        self.layer_stack = nn.Sequential()
        self.layer_stack = model_config.build_linear_layer(layer=self.layer_stack, linear_layers=linear_layers,
                                                           activation=activation,
                                                           dropout_rate=dropout_rate)

    def forward(self, x):
        return self.layer_stack(x)


class Custom_RNN(nn.Module):
    def __init__(self, input_size=11, model_type='LSTM', activation='ReLU', bidirectional=False,
                 hidden_size=64, num_layers=1, linear_layers=None, dropout_rate=0.5, use_cuda=True, cuda_num='cuda:0'):
        super(Custom_RNN, self).__init__()
        self.use_cuda = use_cuda
        self.cuda_num = cuda_num
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.linear_layers = linear_layers
        self.linear_layers.insert(0, hidden_size)
        self.rnn_layer01 = model_config.set_recurrent_layer(name=model_type, input_size=input_size,
                                                            bidirectional=bidirectional,
                                                            hidden_size=hidden_size, num_layers=num_layers)
        self.linear_stack = nn.Sequential()
        self.linear_stack = model_config.build_linear_layer(layer=self.linear_stack, linear_layers=self.linear_layers,
                                                            activation=activation,
                                                            dropout_rate=dropout_rate)

    def forward(self, x):
        output, (h_out, _) = self.rnn_layer01(x, model_config.set_state_h_c(num_layer=self.num_layers, size=x.size(0),
                                                                            hidden_size=self.hidden_size,
                                                                            cuda=self.use_cuda, cuda_num=self.cuda_num))
        return self.linear_stack(output[:, -1, :])


class Custom_CRNN(nn.Module):
    def __init__(self, input_size=11, model_type='LSTM', activation='ReLU', bidirectional=False,
                 hidden_size=64, num_layers=1, linear_layers=None, dropout_rate=0.5, use_cuda=True,
                 convolution_layer=3, use_batch_norm=True, cuda_num='cuda:0'):
        super(Custom_CRNN, self).__init__()
        # convolution layer를 더 쌓는게 맞을지 고민해봐야 할것 같음
        # nn.Conv1d(input_channel, output_channel, kernel_size)\
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


class VanillaNetwork(nn.Module):
    def __init__(self, input_size=11, activation='PReLU', linear=3):
        super(VanillaNetwork, self).__init__()
        self.activation = activation

        if linear == 3:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 64),
                model_config.set_activation(self.activation),
                nn.Linear(64, 64),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(64, 1)
            )
        elif linear == 4:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 32),
                model_config.set_activation(self.activation),
                nn.Linear(32, 64),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(64, 32),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(32, 1)
            )
        elif linear == 5:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 32),
                model_config.set_activation(self.activation),
                nn.Linear(32, 64),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(64, 64),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(64, 32),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(32, 1)
            )

    def forward(self, x):
        return self.linear_relu_stack(x)


class VanillaRecurrentNetwork(nn.Module):
    def __init__(self, input_size=11, recurrent_model='LSTM', activation='PReLU', bidirectional=False,
                 recurrent_hidden_size=64, recurrent_num_layers=1, linear=3, cuda=True):
        super(VanillaRecurrentNetwork, self).__init__()
        self.activation = activation
        self.hidden_size = recurrent_hidden_size
        self.num_layers = recurrent_num_layers
        self.cuda = cuda
        self.recurrent01 = model_config.set_recurrent_layer(name=recurrent_model, input_size=input_size,
                                                            hidden_size=self.hidden_size, num_layers=self.num_layers,
                                                            batch_first=True, bidirectional=bidirectional)
        if linear == 3:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.hidden_size, 64),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(64, 64),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(64, 1)
            )
        elif linear == 4:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.hidden_size, 32),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(32, 64),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(64, 32),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(32, 1)
            )
        elif linear == 5:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.hidden_size, 32),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(32, 64),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(64, 64),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(64, 32),
                nn.Dropout(0.3),
                model_config.set_activation(self.activation),
                nn.Linear(32, 1)
            )

    def forward(self, x):
        if self.cuda:
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        else:
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        output, (h_out, _) = self.recurrent01(x, (h_0, c_0))
        return self.linear_relu_stack(output[:, -1, :])


class VanillaCRNNNetwork(nn.Module):
    def __init__(self, input_size=11, recurrent_model='LSTM', activation='PReLU', bidirectional=False,
                 recurrent_num_layeres=1, recurrent_hidden_size=256):
        super(VanillaCRNNNetwork, self).__init__()
        # convolution
        self.num_layers = recurrent_num_layeres
        self.hidden_size = recurrent_hidden_size
        self.activation = activation
        self.conv1d_layer = nn.Conv1d(input_size, input_size, 3)
        self.lstm_layer = model_config.set_recurrent_layer(name=recurrent_model, input_size=input_size,
                                                           hidden_size=self.hidden_size, num_layers=self.num_layers,
                                                           batch_first=True, bidirectional=bidirectional)
        self.linear_layer1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.3)
        self.activation = model_config.set_activation(self.activation)
        self.linear_layer2 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.conv1d_layer(x)
        # print(out.shape)
        out = out.transpose(1, 2)
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).cuda()
        out, (h_out, _) = self.lstm_layer(out, (h_0, c_0))
        # print('lstm output shape :', out.shape)
        out = self.activation(self.dropout(self.linear_layer1(out[:, -1, :])))
        out = self.linear_layer2(out)
        return out


def model_load(model_configure):
    nn_model = None
    if 'model' in model_configure:
        if model_configure['model'] == 'DNN':
            nn_model = VanillaNetwork(
                model_configure['input_size'],
                activation=model_configure['activation'],
                linear=model_configure['linear']
            )
        elif model_configure['model'] == 'RNN':
            nn_model = VanillaRecurrentNetwork(
                input_size=model_configure['input_size'],
                activation=model_configure['activation'],
                linear=model_configure['linear'],
                cuda=model_configure['cuda']
            )
        elif model_configure['model'] == 'CRNN':
            nn_model = VanillaCRNNNetwork(
                input_size=model_configure['input_size'],
                activation=model_configure['activation']
            )
    else:
        if model_configure['model_type'] == 'Custom_DNN':
            nn_model = Custom_DNN(
                linear_layers=model_configure['linear_layers'],
                activation=model_configure['activation'],
                dropout_rate=model_configure['dropout_rate']
            )
        elif model_configure['model_type'] == 'Custom_RNN':
            nn_model = Custom_RNN(
                input_size=model_configure['input_size'],
                model_type=model_configure['model_type'],
                activation=model_configure['activation'],
                bidirectional=model_configure['bidirectional'],
                hidden_size=model_configure['hidden_size'],
                num_layers=model_configure['num_layers'],
                linear_layers=model_configure['linear_layers'],
                dropout_rate=model_configure['dropout_rate'],
                use_cuda=model_configure['use_cuda']
            )
        elif model_configure['model_type'] == 'Custom_CRNN':
            nn_model = Custom_CRNN(
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

    print("Model structure: ", model, "\n\n")
    return nn_model


if __name__ == '__main__':
    kind = 'Custom_CRNN'
    if kind == 'DNN':
        model = VanillaNetwork().cuda()
    elif kind == 'RNN':
        model = VanillaRecurrentNetwork()
    elif kind == 'CRNN':
        model = VanillaCRNNNetwork().cuda()
    elif kind == 'Custom_DNN':
        model = Custom_DNN(linear_layers=[11, 32, 64, 128, 64, 32, 1]).cuda()
    elif kind == 'Custom_RNN':
        model = Custom_RNN(linear_layers=[32, 64, 128, 64, 32, 1]).cuda()
    elif kind == 'Custom_CRNN':
        model = Custom_CRNN(linear_layers=[32, 64, 128, 64, 32, 1]).cuda()
    print("Model structure: ", model, "\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    # model test
    with torch.no_grad():
        if kind == 'DNN':
            x_data = torch.empty(8,).cuda()
            pred = model(x_data)
            print("pred : ", pred.shape)
        elif kind == 'RNN':
            x_data = torch.empty(1, 8, 8)
            pred = model(x_data)
            print('pred : ', pred.shape)
        elif kind == 'CRNN':
            x_data = torch.empty(1, 11, 15).cuda()
            pred = model(x_data)
            print('pred : ', pred.shape)
        elif kind == 'Custom_DNN':
            x_data = torch.empty(11, ).cuda()
            pred = model(x_data)
            print("pred : ", pred.shape)
        elif kind == 'Custom_RNN':
            x_data = torch.empty(1, 8, 11).cuda()
            pred = model(x_data)
            print('pred : ', pred.shape)
        elif kind == 'Custom_CRNN':
            x_data = torch.empty(2, 11, 15).cuda()
            pred = model(x_data)
            print('pred : ', pred.shape)

