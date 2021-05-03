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


def set_tensorboard_writer(name):
    writer = SummaryWriter(name)
    return writer


def close_tensorboard_writer(writer):
    writer.close()


def close_tensorboard_writer(writer):
    writer.close()


def train(model_config, count, writer_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataloader, \
    test_dataloader, \
    valid_dataloader = data_loader.load_path_loss_with_detail_dataset(input_dir=model_config['input_dir'],
                                                                      device_id=model_config['device_id'],
                                                                      data_environment=model_config['data_environment'],
                                                                      model_type=model_config['model'],
                                                                      scaler=model_config['MinMaxScaler'],
                                                                      use_fspl=model_config['use_fspi'],
                                                                      num_workers=model_config['num_workers'],
                                                                      batch_size=model_config['batch_size'],
                                                                      shuffle=model_config['shuffle'])
    num_epochs = model_config['epoch']
    if model_config['model'] == 'FFNN':
        nn_model = model.VanillaNetwork(input_size).cuda()
        criterion = optimizer.set_criterion(model_config['criterion'])
        optim = optimizer.set_optimizer(model_config['optimizer'],
                                        nn_model, model_config['learning_rate'])
        writer = set_tensorboard_writer('{}/testcase{}'.format(writer_name, str(count).zfill(3)))

        for epoch in range(num_epochs):
            for i, data in enumerate(train_dataloader):
                x_data, y_data = data
                if device:
                    x_data = x_data.cuda()
                    y_data = y_data.cuda()
                y_data = y_data.unsqueeze(-1)
                pred = model(x_data)
                loss = criterion(pred, y_data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ...학습 중 손실(running loss)을 기록하고
                writer.add_scalar('mseloss training loss',
                                  loss / 1000,
                                  epoch * len(train_dataloader) + i)

                if (epoch + 1) % 5 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

            torch.save({epoch: epoch,
                        'model': model,
                        'model_state_dict': model.state_dict()},
                       "checkpoints/testcase{}_epoch_{}.pt".format(str(count).zfill(3), epoch))

            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    total_label = []
                    total_pred = []
                    for i, data in enumerate(test_dataloader):
                        x_data, y_data = data
                        x_data = x_data.cuda()
                        y_data = y_data.cuda()
                        pred = model(x_data)
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
    close_tensorboard_writer(writer)


if __name__ == '__main__':
    data_environment = {'tx_power': 5, 'tx_height': 2, 'rx_height': 0.01, 'tx_antenna_gain': -1.47,
                        'rx_antenna_gain': -1, 'environment': 1}
    model_config = {
        # model config
        'model': 'FFNN', 'criterion': 'MSELoss',
        'optimizer': 'AdaDelta', 'learning_rate': 1,
        'cuda': True, 'batch_size': 128,
        'epoch': 800, 'input_size': 8,
        # dataset config
        'input_dir': '../dataset/v1_timeline',
        'device_id': 'f8:8a:5e:2d:80:f4', 'data_environment': data_environment,
        'use_fspl': True, 'scaler': 'MinMaxScaler',
        'shuffle': True, 'num_workers': 8,
    }

    count = 1
    writer_name = 'runs_prev'

    train(model_config=model_config, count=count, writer_name=writer_name)
