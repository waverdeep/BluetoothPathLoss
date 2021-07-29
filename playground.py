import torch
from model import model
from data import data_loader
import numpy as np
import tool.optimizer as optimizer
import data.data_preprocessing as data_preprocessing
import json
from tool import metrics
import pandas as pd
from tool import use_tensorboard
import random
import tool
# randomness
# random_seed = 42
# np.random.seed(random_seed)
# random.seed(random_seed)
# torch.manual_seed(random_seed)
# 연산속도가 느려질 수 있음
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU


def valid(model_config, nn_model, dataloader, use_cuda, writer, epoch, optimizer, criterion):
    device = torch.device(model_config['cuda_num'])
    with torch.no_grad():
        total_label = []
        total_pred = []
        for i, data in enumerate(dataloader):
            x_data = None
            y_data = None
            if 'model' in model_config:
                if model_config['model'] == 'DNN':
                    x_data, y_data = data
                elif model_config['model'] == 'RNN':
                    x_data = data[:][0]
                    y_data = data[:][1]
                elif model_config['model'] == 'CRNN':
                    x_data = data[:][0]
                    x_data = x_data.transpose(1, 2)
                    y_data = data[:][1]
            else:
                if model_config['model_type'] == 'Custom_DNN':
                    x_data, y_data = data
                elif model_config['model_type'] == 'Custom_RNN':
                    x_data = data[:][0]
                    y_data = data[:][1]
                elif model_config['model_type'] == 'Custom_CRNN':
                    x_data = data[:][0]
                    x_data = x_data.transpose(1, 2)
                    y_data = data[:][1]
            if device:
                x_data = x_data.to(device)
                y_data = y_data.to(device)
            y_pred = nn_model(x_data).reshape(-1)
            loss = criterion(y_pred, y_data)
            writer.add_scalar('Loss/Valid MSELoss', loss / 1000, epoch * len(dataloader) + i)
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


def test(model_config, nn_model, dataloader, device, writer, epoch, criterion):
    with torch.no_grad():
        total_label = []
        total_pred = []
        for i, data in enumerate(dataloader):
            x_data = None
            y_data = None
            if 'model' in model_config:
                if model_config['model'] == 'DNN':
                    x_data, y_data = data
                elif model_config['model'] == 'RNN':
                    x_data = data[:][0]
                    y_data = data[:][1]
                elif model_config['model'] == 'CRNN':
                    x_data = data[:][0]
                    x_data = x_data.transpose(1, 2)
                    y_data = data[:][1]
            else:
                if model_config['model_type'] == 'Custom_DNN':
                    x_data, y_data = data
                elif model_config['model_type'] == 'Custom_RNN':
                    x_data = data[:][0]
                    y_data = data[:][1]
                elif model_config['model_type'] == 'Custom_CRNN':
                    x_data = data[:][0]
                    x_data = x_data.transpose(1, 2)
                    y_data = data[:][1]
            if device:
                x_data = x_data.cuda()
                y_data = y_data.cuda()
            y_pred = nn_model(x_data).reshape(-1)
            loss = criterion(y_pred, y_data)
            writer.add_scalar('Loss/Test MSELoss', loss / 1000, epoch * len(dataloader) + i)
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


def new_train(model_config, count, tb_writer_path, section_message):
    saver = {}
    use_cuda = model_config['use_cuda']
    num_epochs = model_config['epoch']
    sequence_length = 1
    if model_config['model_type'] == 'Custom_RNN' or model_config['model_type'] == 'Custom_CRNN':
        sequence_length = model_config['sequence_length']
    train_dataloader, \
    test_dataloader, \
    valid_dataloader = data_loader.load_path_loss_with_detail_dataset(input_dir=model_config['input_dir'],
                                                                      model_type=model_config['model_type'],
                                                                      num_workers=model_config['num_workers'],
                                                                      batch_size=model_config['batch_size'],
                                                                      shuffle=model_config['shuffle'],
                                                                      input_size=sequence_length)
    nn_model = model.model_load(model_configure=model_config)
    criterion = optimizer.set_criterion(model_config['criterion'])
    if use_cuda:
        criterion = criterion.cuda()
    optim = optimizer.set_optimizer(model_config['optimizer'], nn_model, model_config['learning_rate'])

    if 'scheduler' in model_config:
        scheduler = optimizer.set_scheduler(name=model_config['scheduler'], optim=optim,
                                            lr_lambda=lambda epoch: 0.95 ** epoch)
    writer = use_tensorboard.set_tensorboard_writer('{}/{}_{}'.format(tb_writer_path,
                                                                      section_message,
                                                                      str(count).zfill(3)))
    for epoch in range(num_epochs):
        for i, picked in enumerate(train_dataloader):
            if model_config['model_type'] == 'Custom_DNN':
                x_data, y_data = picked
            elif model_config['model_type'] == 'Custom_RNN':
                x_data = picked[:][0]
                y_data = picked[:][1]
            elif model_config['model_type'] == 'Custom_CRNN':
                x_data = picked[:][0]
                x_data = x_data.transpose(1, 2)
                y_data = picked[:][1]
            if use_cuda:
                x_data = x_data.cuda()
                y_data = y_data.cuda()

            y_pred = nn_model(x_data).reshape(-1)
            loss = criterion(y_pred, y_data)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if 'scheduler' in model_config:
                scheduler.step()
            writer.add_scalar('Loss/Training MSELoss', loss / 1000, epoch * len(train_dataloader) + i)

        if (epoch + 1) % 10 == 0:
            sub_train(model_config=model_config, nn_model=nn_model, dataloader=valid_dataloader, device=use_cuda,
                      writer=writer, epoch=epoch, optimizer=optim, criterion=criterion)
            output = test(model_config=model_config, nn_model=nn_model, dataloader=test_dataloader, device=use_cuda,
                          writer=writer, epoch=epoch, criterion=criterion)
            torch.save({epoch: epoch, 'model': nn_model, 'model_state_dict': nn_model.state_dict()},
                       "{}/{}_{}_epoch_{}.pt".format(checkpoint_dir, message, str(count).zfill(3), epoch))
            saver.update(output)
    use_tensorboard.close_tensorboard_writer(writer)

    saver = pd.DataFrame(saver)
    saver.to_csv('{}/{}_{}.csv'.format(checkpoint_dir, message, str(count).zfill(3)))


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
                                                                                                         shuffle=True,
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
    start_type = 'quick_start' # 'quick_start'

    file_path = 'configurations/v5_v4_part5/'
    checkpoint_dir = 'checkpoints/v5_v4'
    writer_path = 'runs_v5_v4'
    if start_type != 'quick_start':
        checkpoint = None # 'checkpoints_all/CRNN_Adam_LeakyReLU_0.001_sl15_010_epoch_729.pt'
        file_list = tool.file_io.get_all_file_path(file_path, file_extension='json')
        for file in file_list:
            print(file)
            with open(file) as f:
                json_data = json.load(f)
            # filename = data_preprocessing.get_pure_filename(file)
            for idx, data in enumerate(json_data):
                print(idx, data)
                message = "part5_{}_{}_{}_{}_sl{}".format(data['model_type'], data['optimizer'],
                                                            data['activation'], data['learning_rate'],
                                                            data['sequence_length'])
                if 'linear' in data:
                    message += "_{}".format(data['linear'])
                # train(model_config=data, count=idx, writer_path=writer_path, message=message, checkpoint_dir=checkpoint_dir, checkpoint=checkpoint)
                new_train(model_config=data, count=idx, tb_writer_path=writer_path, section_message=message)
    else:
        # part5-01-01
        data = {
            "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
            "convolution_layer": 2, "bidirectional": False, "hidden_size": 64, "num_layers": 1,
            "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
            "use_cuda": True, "batch_size": 30000, "learning_rate": 0.001, "epoch": 2000,
            "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/all",
            "checkpoint_path": "checkpoints/v5_v4/part5_Custom_CRNN_AdamW_ReLU_0.001_sl15_000_epoch_999.pt",
            "scheduler": "LambdaLR"
        }

        print(data)
        message = "part5-01-01_{}_{}_{}_{}_sl{}".format(data['model_type'], data['optimizer'],
                                                        data['activation'], data['learning_rate'],
                                                        data['sequence_length'])
        new_train(model_config=data, count=999, tb_writer_path=writer_path, section_message=message)



########################################################################################################################
# quick01
# data = {
#     "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
#     "convolution_layer": 2, "bidirectional": False, "hidden_size": 256, "num_layers": 2,
#     "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
#     "use_cuda": True, "batch_size": 10000, "learning_rate": 0.0001, "epoch": 1500,
#     "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/new"
# }
# quick02
# data = {
#     "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
#     "convolution_layer": 2, "bidirectional": False, "hidden_size": 512, "num_layers": 1,
#     "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
#     "use_cuda": True, "batch_size": 15000, "learning_rate": 0.0001, "epoch": 1500,
#     "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/new"
# }
# quick03
# data = {
#     "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
#     "convolution_layer": 2, "bidirectional": False, "hidden_size": 512, "num_layers": 2,
#     "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
#     "use_cuda": True, "batch_size": 10000, "learning_rate": 0.0001, "epoch": 1500,
#     "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/new"
# }
# quick04
# data = {
#     "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
#     "convolution_layer": 2, "bidirectional": False, "hidden_size": 256, "num_layers": 1,
#     "linear_layers":[128, 64, 64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
#     "use_cuda": True, "batch_size": 22000, "learning_rate": 0.01, "epoch": 1500,
#     "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/new"
# }
# quick05
# data = {
#     "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
#     "convolution_layer": 2, "bidirectional": False, "hidden_size": 256, "num_layers": 1,
#     "linear_layers": [128, 64, 64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
#     "use_cuda": True, "batch_size": 22000, "learning_rate": 0.01, "epoch": 1500,
#     "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/new"
# }
# quick06-2
# data = {
#     "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
#     "convolution_layer": 1, "bidirectional": False, "hidden_size": 256, "num_layers": 1,
#     "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
#     "use_cuda": True, "batch_size": 10000, "learning_rate": 0.0001, "epoch": 2000,
#     "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/new",
#     "checkpoint_path": "checkpoints/v5_v3/quick06_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_999.pt"
# }
# quick07
# data = {
#     "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
#     "convolution_layer": 2, "bidirectional": False, "hidden_size": 256, "num_layers": 2,
#     "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
#     "use_cuda": True, "batch_size": 20000, "learning_rate": 0.0001, "epoch": 2000,
#     "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/new",
#     "checkpoint_path": "checkpoints/v5_v3/fig2_Custom_CRNN_AdamW_ReLU_0.001_sl15_012_epoch_999.pt"
# }
# quick06-4b
# data = {
#     "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
#     "convolution_layer": 1, "bidirectional": False, "hidden_size": 256, "num_layers": 1,
#     "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
#     "use_cuda": True, "batch_size": 30000, "learning_rate": 0.0001, "epoch": 2000,
#     "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/newb",
#     "checkpoint_path": "checkpoints/v5_v3/quick06-2_Custom_CRNN_AdamW_ReLU_0.0001_sl15_999_epoch_1999.pt"
# }

# quick08
# data = {
#     "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
#     "convolution_layer": 1, "bidirectional": False, "hidden_size": 256, "num_layers": 1,
#     "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
#     "use_cuda": True, "batch_size": 40000, "learning_rate": 0.001, "epoch": 2000,
#     "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/new",
# }

# quick09
# data = {
#     "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "LeakyReLU",
#     "convolution_layer": 2, "bidirectional": False, "hidden_size": 256, "num_layers": 2,
#     "linear_layers": [128, 64, 64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
#     "use_cuda": True, "batch_size": 30000, "learning_rate": 0.01, "epoch": 2000,
#     "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/new",
#     "checkpoint_path": "checkpoints/v5_v3/fig2_Custom_CRNN_AdamW_LeakyReLU_0.001_sl15_031_epoch_629.pt"
# }

# quick06-5
# data = {
#     "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
#     "convolution_layer": 1, "bidirectional": False, "hidden_size": 256, "num_layers": 1,
#     "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.7,
#     "use_cuda": True, "batch_size": 50000, "learning_rate": 0.0001, "epoch": 2000,
#     "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/new",
#     "checkpoint_path": "checkpoints/v5_v3/quick06-2_Custom_CRNN_AdamW_ReLU_0.0001_sl15_999_epoch_1999.pt"
# }

# quick09-1
# data = {
#     "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "LeakyReLU",
#     "convolution_layer": 2, "bidirectional": False, "hidden_size": 256, "num_layers": 2,
#     "linear_layers": [128, 64, 64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
#     "use_cuda": True, "batch_size": 30000, "learning_rate": 0.0005, "epoch": 2000,
#     "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/new",
#     "checkpoint_path": "checkpoints/v5_v3/fig2_Custom_CRNN_AdamW_LeakyReLU_0.001_sl15_031_epoch_999.pt"
# }
