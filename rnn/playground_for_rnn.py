import torch
import torch.nn as nn
import model
from data import data_loader
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorboard as tb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from torch.utils.tensorboard import SummaryWriter
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from sklearn.model_selection import train_test_split


def set_tensorboard_writer(name):
    writer = SummaryWriter(name)
    return writer


def close_tensorboard_writer(writer):
    writer.close()


def close_tensorboard_writer(writer):
    writer.close()


input_sequence = 7
num_epochs = 800
learning_rate = 0.001
batch_size = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_dir = '../dataset/datapack/board/'
file_list = data_loader.get_all_file_path(dataset_dir, file_extension='csv')
train_dataset, test_dataset = train_test_split(file_list, test_size=0.15, shuffle=True, random_state=42)

print('train dataset length : ', len(train_dataset))
print('test dataset length : ', len(test_dataset))

train_dataloader = data_loader.load_pathloss_dataset(train_dataset,
                                                     shuffle=True,
                                                     num_workers=12,
                                                     batch_size=batch_size,
                                                     type='RNN')
test_dataloader = data_loader.load_pathloss_dataset(test_dataset, shuffle=True, batch_size=batch_size,
                                                    num_workers=12, type='RNN')

model = model.VanillaLSTMNetwork(input_size=input_sequence).cuda()

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

writer = set_tensorboard_writer('../runs_rnn/model01-vanilla-sample01')

for epoch in range(num_epochs):
    for i, data in enumerate(train_dataloader):
        y_pred = model(data[:][0].cuda()).reshape(-1)
        y_data = data[:][1].cuda()
        loss = criterion(y_pred, y_data)
        loss.backward()
        optimizer.step()

        # ...학습 중 손실(running loss)을 기록하고
        writer.add_scalar('mseloss training loss',
                          loss / 1000,
                          epoch * len(train_dataloader) + i)

        if (epoch + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            total_label = []
            total_pred = []
            for i, data in enumerate(test_dataloader):
                y_pred = model(data[:][0].cuda()).reshape(-1)
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