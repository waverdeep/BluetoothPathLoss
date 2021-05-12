import torch
import model
from data import data_loader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import tool.optimizer as optimizer
import data.data_preprocessing as data_preprocessing
import json
torch.manual_seed(42)


def set_tensorboard_writer(name):
    writer = SummaryWriter(name)
    return writer


def close_tensorboard_writer(writer):
    writer.close()


def close_tensorboard_writer(writer):
    writer.close()


def test(model_config, nn_model, dataloader, device, writer, epoch):
    with torch.no_grad():
        total_label = []
        total_pred = []
        for i, data in enumerate(dataloader):
            if model_config['model'] == 'FFNN':
                x_data, y_data = data
            elif model_config['model'] == 'RNN':
                x_data = data[:][0]
                y_data = data[:][1]
            if device:
                x_data = x_data.cuda()
            y_pred = nn_model(x_data).reshape(-1).cpu()

            total_label += y_data.tolist()
            total_pred += y_pred.tolist()

        test_mse_score = mean_squared_error(total_label, total_pred)
        test_r2_score = r2_score(total_label, total_pred)
        test_mae_score = mean_absolute_error(total_label, total_pred)
        test_rmse_score = np.sqrt(test_mse_score)

        writer.add_scalar('MSE Score', test_mse_score, epoch)
        writer.add_scalar('R2 Score', test_r2_score, epoch)
        writer.add_scalar('MAE Score', test_mae_score, epoch)
        writer.add_scalar('RMSE Score', test_rmse_score, epoch)

        print("MSE Score : {}".format(test_mse_score))  # 평균제곱 오차가음 낮을수록 좋음
        print("R2 Score : {}".format(test_r2_score))
        print("MAE Score : {}".format(test_mae_score))


def train(model_config, count, writer_name, message, checkpoint_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = model_config['epoch']
    nn_model = None
    train_dataloader, \
    test_dataloader, \
    valid_dataloader = data_loader.load_path_loss_with_detail_dataset(input_dir=model_config['input_dir'],
                                                                      model_type=model_config['model'],
                                                                      num_workers=model_config['num_workers'],
                                                                      batch_size=model_config['batch_size'],
                                                                      shuffle=model_config['shuffle'],
                                                                      input_size=model_config['input_size'] if model_config['model'] == 'FFNN' else model_config['sequence_length'] )
    if model_config['model'] == 'FFNN':
        if 'layer' in model_config:
            nn_model = model.VanillaNetworkCustom(model_config['input_size'], activation=model_config['activation'],
                                                  layer=model_config['layer'])
        else:
            nn_model = model.VanillaNetwork(model_config['input_size'], activation=model_config['activation'])
        if device:
            nn_model = nn_model.cuda()
    elif model_config['model'] == 'RNN' or model_config['model'] == 'CNNRNN':
        if 'recurrent_model' in model_config:
            if 'bidirectional' in model_config:
                nn_model = model.VanillaRecurrentNetwork(model_config['input_size'],
                                                         activation=model_config['activation'],
                                                         recurrent_model=model_config['recurrent_model'],
                                                         bidirectional=model_config['bidirectional'])
            else:
                nn_model = model.VanillaRecurrentNetwork(model_config['input_size'],
                                                         activation=model_config['activation'],
                                                         recurrent_model=model_config['recurrent_model'])
        else:
            nn_model = model.VanillaRecurrentNetwork(model_config['input_size'], activation=model_config['activation'])
        if device:
            nn_model = nn_model.cuda()
    criterion = optimizer.set_criterion(model_config['criterion'])
    if device:
        criterion = criterion.cuda()
    optim = optimizer.set_optimizer(model_config['optimizer'], nn_model, model_config['learning_rate'])
    writer = set_tensorboard_writer('{}/testcase_{}_{}'.format(writer_name, message, str(count).zfill(3)))

    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            if model_config['model'] == 'FFNN':
                x_data, y_data = data
            elif model_config['model'] == 'RNN':
                x_data = data[:][0]
                y_data = data[:][1]
            elif model_config['model'] == 'CNNRNN':
                x_data = data[:][0]
                x_data = x_data.transpose(1, 2)
                y_data = data[:][1]
            if device:
                x_data = x_data.cuda()
                y_data = y_data.cuda()
            y_pred = nn_model(x_data).reshape(-1)
            loss = criterion(y_pred, y_data)
            optim.zero_grad()
            loss.backward()
            optim.step()
            # ...학습 중 손실(running loss)을 기록하고
            writer.add_scalar('mseloss training loss', loss / 1000, epoch * len(train_dataloader) + i)
            torch.save({epoch: epoch, 'model': nn_model, 'model_state_dict': nn_model.state_dict()},
                       "{}/testcase_{}_{}_epoch_{}.pt".format(checkpoint_dir, message, str(count).zfill(3), epoch))

        if (epoch + 1) % 10 == 0:
            test(model_config=model_config, nn_model=nn_model, dataloader=test_dataloader, device=device, writer=writer, epoch=epoch)
    close_tensorboard_writer(writer)


if __name__ == '__main__':
    file_path = 'configurations_v2/'
    writer_name = 'runs_v2'
    checkpoint_dir = 'checkpoints_v2'
    file_list = data_preprocessing.get_all_file_path(file_path, file_extension='json')
    print(file_list[8:])
    file_list = file_list[8:]

    for file in file_list:
        with open(file) as f:
            json_data = json.load(f)
        filename = data_preprocessing.get_pure_filename(file)
        for idx, data in enumerate(json_data):
            print(idx, data)

            train(model_config=data, count=idx, writer_name=writer_name, message=filename, checkpoint_dir=checkpoint_dir)


