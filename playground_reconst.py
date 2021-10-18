import torch
from tool import use_tensorboard
import data.data_loader_reconst as data_loader
import tool.optimizer as optimizer
import tool.metrics as metrics
from model_pack import model_reconst
from tqdm import tqdm
import numpy as np


def main(config):
    device = None
    if config['use_cuda']:
        device = torch.device(config['cuda_num'])

    print(">>> load train, valid, test dataloader <<<")
    train_dataloader, valid_dataloader, test_dataloader = data_loader.load_path_loss_with_detail_dataset(
        input_dir=config['input_dir'],
        model_type=config['model_type'],
        num_workers=config['num_workers'],
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        input_size=config['sequence_length']
    )

    print(">>> setup tensorboard <<<")
    writer = use_tensorboard.set_tensorboard_writer(
        '{}/{}'.format(config['tensorboard_writer_path'],
                       config['section_message'])
    )

    print(">>> model load <<<")
    model = model_reconst.model_load(model_configure=config)

    print(">>> loss, optimizer setup <<<")
    criterion = optimizer.set_criterion(config['criterion'])
    optim = optimizer.set_optimizer(config['optimizer'], model=model, learning_rate=config['learning_rate'])

    num_of_epoch = config['epoch']
    for epoch in range(num_of_epoch):
        print("start training ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        train(train_loader=train_dataloader,
              epoch=epoch,
              config=config,
              device=device,
              model=model,
              criterion=criterion,
              writer=writer,
              optim=optim)
        validation(valid_loader=valid_dataloader,
                   epoch=epoch,
                   config=config,
                   device=device,
                   model=model,
                   criterion=criterion,
                   writer=writer)
        torch.save({"epoch": epoch,
                    "model": model,
                    "model_state_dict": model.state_dict()
                    }, "{}/{}_epoch_{}.pt".format(config['checkpoint_dir'], config['section_message'], epoch))


def train(train_loader, epoch, config, device, model, criterion, writer, optim):
    # train_bar = tqdm(train_loader,
    #                  desc='{}/{} epoch train ... '.format(epoch, config['epoch']))
    for batch_idx, data in enumerate(train_loader):
        x_data = data[:][0].transpose(1, 2)
        # print(x_data.size())
        y_data = data[:][1]

        if config['use_cuda']:
            x_data = x_data.to(device)
            y_data = y_data.to(device)

        # 모델 예측 진행
        y_pred = model(x_data).reshape(-1)
        # 예측결과에 대한 Loss 계산
        loss = criterion(y_pred, y_data)
        # 역전파 수행
        optim.zero_grad()
        loss.backward()
        optim.step()

        writer.add_scalar("Loss/Train MSELoss", loss/1000, epoch * len(train_loader) + batch_idx)
        y_pred = y_pred.cpu().detach().numpy()
        y_data = y_data.cpu().detach().numpy()

        mse_score = metrics.mean_squared_error(y_data, y_pred)
        r2_score = metrics.r2_score(y_data, y_pred)
        mae_score = metrics.mean_absolute_error(y_data, y_pred)
        rmse_score = np.sqrt(mse_score)
        mape_score = metrics.mean_absolute_percentage_error(y_data, y_pred)

        writer.add_scalar('MSE Score/train', mse_score, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('R2 Score/train', r2_score, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('MAE Score/train', mae_score, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('RMSE Score/train', rmse_score, epoch * len(train_loader) + batch_idx)
        writer.add_scalar('MAPE Score/train', mape_score, epoch * len(train_loader) + batch_idx)

        # train_bar.set_description('{}/{} epoch [ MAE: {}/ loss: {} ]'.format(
        #     epoch, config['epoch'], round(float(mae_score), 3), round(float(loss/1000), 3)))

        conv = 0
        for idx, layer in enumerate(model.modules()):
            if isinstance(layer, torch.nn.Conv1d):
                writer.add_histogram("Conv/weights-{}".format(conv), layer.weight,
                                     global_step=(epoch - 1) * len(train_loader) + batch_idx)
                writer.add_histogram("Conv/bias-{}".format(conv), layer.bias,
                                     global_step=(epoch - 1) * len(train_loader) + batch_idx)
                conv += 1
            if isinstance(layer, torch.nn.LSTM):
                writer.add_histogram("GRU/weight_ih_l0",
                                     layer.weight_ih_l0, global_step=(epoch - 1) * len(train_loader) + batch_idx)
                writer.add_histogram("GRU/weight_hh_l0",
                                     layer.weight_hh_l0, global_step=(epoch - 1) * len(train_loader) + batch_idx)
                writer.add_histogram("GRU/bias_ih_l0",
                                     layer.bias_ih_l0, global_step=(epoch - 1) * len(train_loader) + batch_idx)
                writer.add_histogram("GRU/bias_hh_l0",
                                     layer.bias_hh_l0, global_step=(epoch - 1) * len(train_loader) + batch_idx)


def validation(valid_loader, epoch, config, device, model, criterion, writer):
    with torch.no_grad():
        total_label = []
        total_pred = []
        # train_bar = tqdm(valid_loader,
        #                  desc='{}/{} epoch train ... '.format(epoch, config['epoch']))
        for batch_idx, data in enumerate(valid_loader):
            x_data = data[:][0].transpose(1, 2)
            y_data = data[:][1]

            if config['use_cuda']:
                x_data = x_data.to(device)
                y_data = y_data.to(device)

            # 모델 예측 진행
            y_pred = model(x_data).reshape(-1)
            # 예측결과에 대한 Loss 계산
            loss = criterion(y_pred, y_data)

            writer.add_scalar('Loss/Validation MSELoss', loss / 1000, epoch * len(valid_loader) + batch_idx)

            y_pred = y_pred.cpu()

            total_label += y_data.tolist()
            total_pred += y_pred.tolist()
        mse_score = metrics.mean_squared_error(total_label, total_pred)
        r2_score = metrics.r2_score(total_label, total_pred)
        mae_score = metrics.mean_absolute_error(total_label, total_pred)
        rmse_score = np.sqrt(mse_score)
        mape_score = metrics.mean_absolute_percentage_error(total_label, total_pred)

        writer.add_scalar('MSE Score/Validation', mse_score, epoch)
        writer.add_scalar('R2 Score/Validation', r2_score, epoch)
        writer.add_scalar('MAE Score/Validation', mae_score, epoch)
        writer.add_scalar('RMSE Score/Validation', rmse_score, epoch)
        writer.add_scalar('MAPE Score/Validation', mape_score, epoch)


def test():
    pass


if __name__ == '__main__':
    # CRNN-VSeq-ReLU-AdamW-T01
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "ReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdamW",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-AdamW-T01",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-ReLU-Adam-T02
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "ReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "Adam",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.0005,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-Adam-T02",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-ReLU-Adam-T03
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "ReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "Adam",
        "dropout_rate": 0.7,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-Adam-T03",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-ReLU-AdamW-T04
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "ReLU",
        "convolution_layer": 3,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdamW",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-AdamW-T04",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-ReLU-AdamW-T05
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "ReLU",
        "convolution_layer": 3,
        "bidirectional": False,
        "hidden_size": 128,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdamW",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 512,
        "learning_rate": 0.005,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-AdamW-T05",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"
    }
    # CRNN-VSeq-ReLU-AdaDelta-T06
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "ReLU",
        "convolution_layer": 3,
        "bidirectional": False,
        "hidden_size": 128,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdaDelta",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.005,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-AdaDelta-T06",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"
    }
    # CRNN-VSeq-ReLU-Adam-T07
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "ReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 512,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "Adam",
        "dropout_rate": 0.7,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-Adam-T07",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-ReLU-AdaGrad-T08
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "ReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdaGrad",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-AdaGrad-T08",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-ReLU-SGD-T09
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "ReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "SGD",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-SGD-T09",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-ReLU-SGD-T10
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "ReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 512,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "SGD",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-SGD-T10",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-ReLU-SGD-T11
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "ReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 512,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "SGD",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.01,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-SGD-T11",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-ReLU-SGD-T12
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "ReLU",
        "convolution_layer": 4,
        "bidirectional": False,
        "hidden_size": 512,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "SGD",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.005,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-SGD-T12",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-LeakyReLU-AdamW-T13
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "LeakyReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdamW",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-LeakyReLU-AdamW-T13",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-PReLU-AdamW-T14
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "PReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdamW",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-PReLU-AdamW-T14",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-PReLU-AdamW-T15
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "PReLU",
        "convolution_layer": 4,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdamW",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 500,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-PReLU-AdamW-T15",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-PReLU-AdamW-T16
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "PReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 512,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdamW",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 500,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-PReLU-AdamW-T16",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-PReLU-AdamW-T17
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "PReLU",
        "convolution_layer": 4,
        "bidirectional": False,
        "hidden_size": 512,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdamW",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 500,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-PReLU-AdamW-T17",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-PReLU-AdaDelta-T18
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "PReLU",
        "convolution_layer": 4,
        "bidirectional": False,
        "hidden_size": 512,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdaDelta",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.05,
        "epoch": 500,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-PReLU-AdaDelta-T18",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-PReLU-AdaDelta-T19
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "PReLU",
        "convolution_layer": 4,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 32, 1],
        "criterion": "MSELoss",
        "optimizer": "AdaDelta",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.005,
        "epoch": 500,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-PReLU-AdaDelta-T19",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-PReLU-AdamW-T20
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 10,
        "activation": "PReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdamW",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-PReLU-AdamW-T20",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    # CRNN-VSeq-PReLU-AdamW-T21
    configuration = {
        "model_type": "CRNN",
        "input_size": 11,
        "sequence_length": 18,
        "activation": "PReLU",
        "convolution_layer": 2,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "AdamW",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 100,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-PReLU-AdamW-T21",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }

    # CRNN-VSeq-PReLU-AdamW-T24
    configuration = {
        "model_type": "DilatedCRNN",
        "input_size": 11,
        "sequence_length": 15,
        "activation": "PReLU",
        "convolution_layer": 4,
        "bidirectional": False,
        "hidden_size": 256,
        "num_layers": 1,
        "linear_layers": [64, 1],
        "criterion": "MSELoss",
        "optimizer": "Adam",
        "dropout_rate": 0.5,
        "use_cuda": True,
        "cuda_num": "cuda:0",
        "batch_size": 1024,
        "learning_rate": 0.001,
        "epoch": 500,
        "num_workers": 8,
        "shuffle": True,
        "input_dir": "dataset/v7_all/point_all_v6",
        "tensorboard_writer_path": "runs_2021_10_15",
        "section_message": "CRNN-VSeq-ReLU-Adam-T25",
        "checkpoint_dir": "checkpoints/2021_10_15"
        # "checkpoint_path": "checkpoints/2021_09_16/quick_02-03_Custom_CRNN_AdamW_ReLU_0.001_sl15_999_epoch_1999.pt"

    }
    print(">>> Training Bluetooth PathLoss <<<")
    print("train configuration ->->->")
    print(configuration)
    main(config=configuration)
