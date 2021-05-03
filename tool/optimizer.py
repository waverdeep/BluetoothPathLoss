import torch


def set_optimizer(name, model, learning_rate):
    if name == 'AdaDelta':
        return torch.optim.Adadelta(model.parameter(), learning_rate)
    elif name == 'AdaGrad':
        return torch.optim.AdaGrad(model.parameter(), learning_rate)
    elif name == 'Adam':
        return torch.optim.Adam(model.parameter(), learning_rate)
    elif name == 'AdamW':
        return torch.optim.AdamW(model.parameter(), learning_rate)
    elif name == 'SGD':
        return torch.optim.SGD(model.parameter(), learning_rate)


def set_criterion(name):
    if name == 'MSELoss':
        return torch.nn.MSELoss()
    elif name == 'L1Loss':
        return torch.nn.L1Loss()
