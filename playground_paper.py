import torch
from tool import use_tensorboard
import tool.optimizer as optimizer
import data.data_loader_scaler as data_loader
import tool.metrics as metrics
from model_pack import model_paper
import numpy as np
import matplotlib.pyplot as plt


def main(config):
    device = None
    if config['use_cuda']:
        device = torch.device(config['cuda_num'])

    print(">>> Load train, valid dataset <<<")
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
    model = model_paper.model_load(model_configure=config)

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
    for batch_idx, data in enumerate(train_loader):
        x_data = data[:][0].transpose(1, 2)
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


def validation(valid_loader, epoch, config, device, model, criterion, writer):
    with torch.no_grad():
        total_label = []
        total_pred = []
        total_x = []
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
            for temp in data[:][0]:
                x = temp[:, 0].cpu()
                total_x.append(np.array(x.tolist()).mean())
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
        fig = plt.figure(figsize=(24, 16))
        plt.scatter(total_x, total_pred, color='blue', alpha=0.2, label='prediction')
        plt.scatter(total_x, total_label, color='red', alpha=0.2, label='groundtruth')
        plt.legend()
        plt.grid(True)
        plt.xlabel("rssi (dbm)")
        plt.ylabel("distance (meter)")
        plt.title("Prediction Result")
        plt.yticks(np.arange(0, 70, 5))
        writer.add_figure('PathLoss Prediction', fig, epoch)

        data_size = int(len(total_x) / 16)
        fig_detail = plt.figure(figsize=(16, 16))
        plt.subplots(constrained_layout=True)
        for i in range(16):
            plt.subplot(4, 4, 1 + i)
            if i < 15:
                plt.scatter(total_x[data_size * i: data_size * (i + 1)], total_pred[data_size * i: data_size * (i + 1)], color='blue', alpha=0.2, label='prediction')
                plt.scatter(total_x[data_size * i: data_size * (i + 1)], total_label[data_size * i: data_size * (i + 1)], color='red', alpha=0.2, label='groundtruth')
                plt.legend()
                plt.grid(True)
                plt.xlabel("rssi (dbm)")
                plt.ylabel("distance (meter)")
                plt.yticks(np.arange(0, 70, 5))
                plt.title("PathLoss Prediction with Detail")
            else:
                plt.scatter(total_x[data_size * i:], total_pred[data_size * i:], color='blue', alpha=0.2, label='prediction')
                plt.scatter(total_x[data_size * i:], total_label[data_size * i:], color='red', alpha=0.2, label='groundtruth')
                plt.legend()
                plt.grid(True)
                plt.xlabel("rssi (dbm)")
                plt.ylabel("distance (meter)")
                plt.yticks(np.arange(0, 70, 5))
                plt.title("PathLoss Prediction with Detail")
        plt.subplots(constrained_layout=True)
        writer.add_figure('VisualizeValidationDetail', fig_detail, epoch)


if __name__ == '__main__':
    configuration = {

    }

    print(">>> Training Bluetooth PathLoss <<<")
    print("train configuration ->->->")
    print(configuration)
    main(config=configuration)