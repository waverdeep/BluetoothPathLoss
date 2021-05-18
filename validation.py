import torch
from data import data_loader
import model
from tool import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np

# gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(model_config, nn_model, dataloader, device):
    with torch.no_grad():
        total_rssi = []
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
            x_data = x_data.cpu()
            y_data = y_data.cpu()
            total_rssi += x_data[:, 0].tolist()
            total_label += y_data.tolist()
            total_pred += y_pred.tolist()

        test_mse_score = mean_squared_error(total_label, total_pred)
        test_r2_score = r2_score(total_label, total_pred)
        test_mae_score = mean_absolute_error(total_label, total_pred)
        test_rmse_score = np.sqrt(test_mse_score)





if __name__ == '__main__':
    # parameters
    input_size = 8
    batch_size = 64
    checkpoint_path = 'checkpoints/model_adadelta_mse_epoch_529.pt'
    valid_data_dir = 'dataset/pathloss_v1_valid_cs.csv'

    configure = {
    "model": "RNN",
    "criterion": "MSELoss",
    "optimizer": "AdamW","activation": "ReLU",
    "learning_rate": 0.001,
    "cuda": True,
    "batch_size": 512,
    "epoch": 800,
    "input_size": 8,
    "sequence_length": 15,
    "input_dir": "dataset/v1_scaled_mm",
    "shuffle": True,
    "num_workers": 8
  }
    test(model_config=model_config, nn_model=nn_model, dataloader=test_dataloader, device=device, writer=writer,
         epoch=epoch)