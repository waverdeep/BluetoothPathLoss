import torch


def set_optimizer(name, model, learning_rate):
    if name == 'AdaDelta':
        return torch.optim.Adadelta(model.parameters(), learning_rate)
    elif name == 'AdaGrad':
        return torch.optim.Adagrad(model.parameters(), learning_rate)
    elif name == 'Adam':
        return torch.optim.Adam(model.parameters(), learning_rate)
    elif name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), learning_rate)
    elif name == 'SGD':
        return torch.optim.SGD(model.parameters(), learning_rate)


def set_criterion(name):
    if name == 'MSELoss':
        return torch.nn.MSELoss()
    elif name == 'L1Loss':
        return torch.nn.L1Loss()


# https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
def set_scheduler(name, optim, lr_lambda):
    if name == 'LambdaLR':
        return torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lr_lambda)
    elif name == 'MultiplicativeLR':
        pass
    elif name == 'StepLR':
        pass
    elif name == 'MultiStepLR':
        pass
    elif name == 'ExponentialLR':
        pass


