import torch
import model
from data import data_loader
import numpy as np
import tool.optimizer as optimizer
import data.data_preprocessing as data_preprocessing
import json
from tool import metrics
torch.manual_seed(42)
import pandas as pd
from tool import use_tensorboard

from ast import literal_eval


def sub_train(model_config, nn_model, dataloader, device, writer, epoch, optimizer, criterion):
    total_label = []
    total_pred = []
    for i, data in enumerate(dataloader):
        x_data = None
        y_data = None
        if model_config['model'] == 'DNN':
            x_data, y_data = data
        elif model_config['model'] == 'RNN':
            x_data = data[:][0]
            y_data = data[:][1]
        elif model_config['model'] == 'CRNN':
            x_data = data[:][0]
            x_data = x_data.transpose(1, 2)
            y_data = data[:][1]
        if device:
            x_data = x_data.cuda()
            y_data = y_data.cuda()
        y_pred = nn_model(x_data).reshape(-1)
        optimizer.zero_grad()
        loss = criterion(y_pred, y_data)
        loss.backward()
        optimizer.step()
        y_pred = y_pred.cpu()

        total_label += y_data.tolist()
        total_pred += y_pred.tolist()

    test_mse_score = metrics.mean_squared_error(total_label, total_pred)
    test_r2_score = metrics.r2_score(total_label, total_pred)
    test_mae_score = metrics.mean_absolute_error(total_label, total_pred)
    test_rmse_score = np.sqrt(test_mse_score)
    test_mape_score = metrics.mean_absolute_percentage_error(total_label, total_pred)

    writer.add_scalar('MSE Score/sub_train', test_mse_score, epoch)
    writer.add_scalar('R2 Score/sub_train', test_r2_score, epoch)
    writer.add_scalar('MAE Score/sub_train', test_mae_score, epoch)
    writer.add_scalar('RMSE Score/sub_train', test_rmse_score, epoch)
    writer.add_scalar('MAPE Score/sub_train', test_mape_score, epoch)


def test(model_config, nn_model, dataloader, device, writer, epoch):
    with torch.no_grad():
        total_label = []
        total_pred = []
        for i, data in enumerate(dataloader):
            x_data = None
            y_data = None
            if model_config['model'] == 'DNN':
                x_data, y_data = data
            elif model_config['model'] == 'RNN':
                x_data = data[:][0]
                y_data = data[:][1]
            elif model_config['model'] == 'CRNN':
                x_data = data[:][0]
                x_data = x_data.transpose(1, 2)
                y_data = data[:][1]
            if device:
                x_data = x_data.cuda()
            y_pred = nn_model(x_data).reshape(-1)
            y_pred = y_pred.cpu()

            total_label += y_data.tolist()
            total_pred += y_pred.tolist()
        test_mse_score = metrics.mean_squared_error(total_label, total_pred)
        test_r2_score = metrics.r2_score(total_label, total_pred)
        test_mae_score = metrics.mean_absolute_error(total_label, total_pred)
        test_rmse_score = np.sqrt(test_mse_score)
        test_mape_score = metrics.mean_absolute_percentage_error(total_label, total_pred)

        writer.add_scalar('MSE Score/test', test_mse_score, epoch)
        writer.add_scalar('R2 Score/test', test_r2_score, epoch)
        writer.add_scalar('MAE Score/test', test_mae_score, epoch)
        writer.add_scalar('RMSE Score/test', test_rmse_score, epoch)
        writer.add_scalar('MAPE Score/test', test_mape_score, epoch)
        print('===epoch {}==='.format(epoch))
        print("MSE Score : {}".format(test_mse_score))  # 평균제곱 오차가음 낮을수록 좋음
        print("R2 Score : {}".format(test_r2_score))
        print("MAE Score : {}".format(test_mae_score))

        return {epoch + 1: [test_mae_score, test_mse_score, test_rmse_score, test_r2_score]}


def train(model_config, count, writer_path, message, checkpoint_dir, checkpoint=None):
    saver = {}
    device = model_config['cuda']
    num_epochs = model_config['epoch']
    if model_config['model'] == 'DNN':
        size = model_config['input_size']
    elif model_config['model'] == 'RNN' or model_config['model'] == 'CRNN':
        size = model_config['sequence_length']
    train_dataloader, test_dataloader, valid_dataloader = data_loader.load_path_loss_with_detail_dataset(input_dir=model_config['input_dir'],
                                                                                                         model_type=model_config['model'],
                                                                                                         num_workers=model_config['num_workers'],
                                                                                                         batch_size=model_config['batch_size'],
                                                                                                         shuffle=model_config['shuffle'],
                                                                                                         input_size=size)
    nn_model = model.model_load(model_configure=model_config)
    if checkpoint is not None:
        prev_checkpoint = torch.load(checkpoint)
        nn_model.load_state_dict(prev_checkpoint['model_state_dict'])
    criterion = optimizer.set_criterion(model_config['criterion'])
    if device:
        criterion = criterion.cuda()
    optim = optimizer.set_optimizer(model_config['optimizer'], nn_model, model_config['learning_rate'])
    writer = use_tensorboard.set_tensorboard_writer('{}/{}_{}'.format(writer_path, message, str(count).zfill(3)))

    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            if model_config['model'] == 'DNN':
                x_data, y_data = data
            elif model_config['model'] == 'RNN':
                x_data = data[:][0]
                y_data = data[:][1]
            elif model_config['model'] == 'CRNN':
                x_data = data[:][0]
                x_data = x_data.transpose(1, 2)
                y_data = data[:][1]
            if device:
                x_data = x_data.cuda()
                y_data = y_data.cuda()
            y_pred = nn_model(x_data).reshape(-1)
            loss = criterion(y_pred, y_data)
            # print('y_pred : ', y_pred.cpu())
            # print('y_data : ', y_data.cpu())
            optim.zero_grad()
            loss.backward()
            optim.step()
            # ...학습 중 손실(running loss)을 기록하고
            writer.add_scalar('Training MSELoss', loss / 1000, epoch * len(train_dataloader) + i)

        if (epoch + 1) % 10 == 0:
            sub_train(model_config=model_config, nn_model=nn_model, dataloader=valid_dataloader, device=device,
                      writer=writer, epoch=epoch, optimizer=optim, criterion=criterion)
            output = test(model_config=model_config, nn_model=nn_model, dataloader=test_dataloader, device=device,
                          writer=writer, epoch=epoch)
            torch.save({epoch: epoch, 'model': nn_model, 'model_state_dict': nn_model.state_dict()},
                       "{}/{}_{}_epoch_{}.pt".format(checkpoint_dir, message, str(count).zfill(3), epoch))
            saver.update(output)
    use_tensorboard.close_tensorboard_writer(writer)

    saver = pd.DataFrame(saver)
    saver.to_csv('{}/{}_{}.csv'.format(checkpoint_dir, message, str(count).zfill(3)))


if __name__ == '__main__':
    file_path = 'old/configurations_old/configurations_v3'
    checkpoint_dir = 'old/configurations_old/type_data_size/checkpoints_size_50'
    writer_path = 'runs_size_50'
    checkpoint = None # 'checkpoints_all/CRNN_Adam_LeakyReLU_0.001_sl15_010_epoch_729.pt'
    file_list = data_preprocessing.get_all_file_path(file_path, file_extension='json')
    for file in file_list:
        print(file)
        with open(file) as f:
            json_data = json.load(f)
        # filename = data_preprocessing.get_pure_filename(file)
        for idx, data in enumerate(json_data):
            print(idx, data)
            message = "{}_{}_{}_{}_sl{}".format(data['model'], data['optimizer'],
                                                        data['activation'], data['learning_rate'],
                                                        data['sequence_length'])
            train(model_config=data, count=idx, writer_path=writer_path, message=message, checkpoint_dir=checkpoint_dir, checkpoint=checkpoint)


