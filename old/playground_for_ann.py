import torch
import torch.nn as nn
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
from tool import optimizer
from data import data_preprocessing
import json


def set_tensorboard_writer(name):
    writer = SummaryWriter(name)
    return writer


def close_tensorboard_writer(writer):
    writer.close()


def close_tensorboard_writer(writer):
    writer.close()


def train(model_config, count, writer_name, message):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_config['model'] == 'FFNN':
        train_dataloader, \
        test_dataloader, \
        valid_dataloader = data_loader.load_path_loss_with_detail_dataset(input_dir=model_config['input_dir'],
                                                                          model_type=model_config['model'],
                                                                          num_workers=model_config['num_workers'],
                                                                          batch_size=model_config['batch_size'],
                                                                          shuffle=model_config['shuffle'],
                                                                          input_size=model_config['input_size'])
    elif model_config['model'] == 'RNN':
        train_dataloader, \
        test_dataloader, \
        valid_dataloader = data_loader.load_path_loss_with_detail_dataset(input_dir=model_config['input_dir'],
                                                                          model_type=model_config['model'],
                                                                          num_workers=model_config['num_workers'],
                                                                          batch_size=model_config['batch_size'],
                                                                          shuffle=model_config['shuffle'],
                                                                          input_size=model_config['sequence_length'])
    num_epochs = model_config['epoch']
    if model_config['model'] == 'FFNN':
        nn_model = model.VanillaNetwork(model_config['input_size'], activation=model_config['activation']).cuda()
        criterion = optimizer.set_criterion(model_config['criterion']).cuda()
        optim = optimizer.set_optimizer(model_config['optimizer'],
                                        nn_model, model_config['learning_rate'])
        writer = set_tensorboard_writer('{}/testcase_{}_{}'.format(writer_name, message, str(count).zfill(3)))

        for epoch in range(num_epochs):
            for i, data in enumerate(train_dataloader):
                x_data, y_data = data
                if device:
                    x_data = x_data.cuda()
                    y_data = y_data.cuda()
                y_data = y_data.unsqueeze(-1)
                pred = nn_model(x_data)
                loss = criterion(pred, y_data)
                optim.zero_grad()
                loss.backward()
                optim.step()

                # ...학습 중 손실(running loss)을 기록하고
                writer.add_scalar('mseloss training loss',
                                  loss / 1000,
                                  epoch * len(train_dataloader) + i)

                if (epoch + 1) % 5 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

            torch.save({epoch: epoch,
                        'model': nn_model,
                        'model_state_dict': nn_model.state_dict()},
                       "../checkpoints_v1/testcase_{}_{}_epoch_{}.pt".format(message, str(count).zfill(3), epoch))

            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    total_label = []
                    total_pred = []
                    for i, data in enumerate(test_dataloader):
                        x_data, y_data = data
                        x_data = x_data.cuda()
                        y_data = y_data.cuda()
                        pred = nn_model(x_data)
                        pred = pred.squeeze(-1)
                        rssi = x_data[:, 0]
                        rssi = rssi.cpu().numpy()
                        y_data = y_data.cpu().numpy()
                        pred = pred.cpu().numpy()

                        total_label += y_data.tolist()
                        total_pred += pred.tolist()

                    test_mse_score = mean_squared_error(total_label, total_pred)
                    test_r2_score = r2_score(total_label, total_pred)
                    test_mae_score = mean_absolute_error(total_label, total_pred)
                    test_rmse_score = np.sqrt(test_mse_score)

                    writer.add_scalar('MSE Score',
                                      test_mse_score,
                                      epoch)
                    writer.add_scalar('R2 Score',
                                      test_r2_score,
                                      epoch)
                    writer.add_scalar('MAE Score',
                                      test_mae_score,
                                      epoch)
                    writer.add_scalar('RMSE Score',
                                      test_rmse_score,
                                      epoch)

                    print("MSE Score : {}".format(test_mse_score))  # 평균제곱 오차가음 낮을수록 좋음
                    print("R2 Score : {}".format(test_r2_score))
                    print("MAE Score : {}".format(test_mae_score))

    elif model_config['model'] == 'RNN':
        nn_model = model.VanillaRecurrentNetwork(model_config['input_size'], activation=model_config['activation']).cuda()
        criterion = optimizer.set_criterion(model_config['criterion']).cuda()
        optim = optimizer.set_optimizer(model_config['optimizer'],
                                        nn_model, model_config['learning_rate'])
        writer = set_tensorboard_writer('{}/testcase_{}_{}'.format(writer_name, message, str(count).zfill(3)))

        for epoch in range(num_epochs):
            for i, data in enumerate(train_dataloader):
                y_pred = nn_model(data[:][0].cuda()).reshape(-1)
                y_data = data[:][1].cuda()
                loss = criterion(y_pred, y_data)
                # print('x_data: ', data[:][0])
                # print('x_data shape: ', data[:][0].shape)
                # print('y_data: ', y_data)
                # print('y_pred: ', y_pred)
                loss.backward()
                optim.step()

                # ...학습 중 손실(running loss)을 기록하고
                writer.add_scalar('mseloss training loss',
                                  loss / 1000,
                                  epoch * len(train_dataloader) + i)

                if (epoch + 1) % 5 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

                torch.save({epoch: epoch,
                            'model': nn_model,
                            'model_state_dict': nn_model.state_dict()},
                           "../checkpoints_prev/testcase_{}_{}_epoch_{}.pt".format(message, str(count).zfill(3), epoch))

            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    total_label = []
                    total_pred = []
                    for i, data in enumerate(test_dataloader):
                        y_pred = nn_model(data[:][0].cuda()).reshape(-1)
                        y_pred = y_pred.cpu()
                        # rssi = x_data[:, 0]
                        # rssi = rssi.cpu().numpy()
                        y_data = data[:][1].numpy()
                        y_pred = y_pred.numpy()
                        # print('y_data : ', y_data)
                        # print('y_pred : ', y_pred)

                        total_label += y_data.tolist()
                        total_pred += y_pred.tolist()

                    test_mse_score = mean_squared_error(total_label, total_pred)
                    test_r2_score = r2_score(total_label, total_pred)
                    test_mae_score = mean_absolute_error(total_label, total_pred)
                    test_rmse_score = np.sqrt(test_mse_score)

                    writer.add_scalar('MSE Score',
                                      test_mse_score,
                                      epoch)
                    writer.add_scalar('R2 Score',
                                      test_r2_score,
                                      epoch)
                    writer.add_scalar('MAE Score',
                                      test_mae_score,
                                      epoch)
                    writer.add_scalar('RMSE Score',
                                      test_rmse_score,
                                      epoch)

                    print("MSE Score : {}".format(test_mse_score))  # 평균제곱 오차가음 낮을수록 좋음
                    print("R2 Score : {}".format(test_r2_score))
                    print("MAE Score : {}".format(test_mae_score))

    close_tensorboard_writer(writer)


if __name__ == '__main__':
    model_config = {
        # model config
        'model': 'RNN', 'criterion': 'MSELoss',
        'optimizer': 'AdaDelta', 'learning_rate': 0.001,
        'cuda': True, 'batch_size': 256,
        'epoch': 800, 'input_size': 8, 'sequence_length': 10,
        # dataset config
        'input_dir': '../dataset/v1_scaled',
        'shuffle': True, 'num_workers': 8,
    }

    file_list = data_preprocessing.get_all_file_path('../configures/configurations_v1/', file_extension='json')
    for file in file_list:
        with open(file) as f:
            json_data = json.load(f)
        filename = data_preprocessing.get_pure_filename(file)
        for idx, data in enumerate(json_data):
            print(idx, data)
            writer_name = '../runs_v1'
            train(model_config=data, count=idx, writer_name=writer_name, message=filename)

