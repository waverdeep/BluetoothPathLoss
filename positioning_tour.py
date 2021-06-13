import torch
import data.data_preprocessing as preprocessing
import model
import pandas as pd
import numpy as np
from data import data_loader
from scipy.stats import mode

model_configure = {"model": "CRNN", "criterion": "MSELoss","optimizer": "Adam","activation": "LeakyReLU",
                   "learning_rate": 0.001, "cuda": True, "batch_size": 512, "epoch": 800, "input_size": 8,
                   "sequence_length": 15, "input_dir": "dataset/v1_scaled_mm", "shuffle": True, "num_workers": 8,
                   'checkpoint_path': 'checkpoints_all_v3/CRNN_Adam_LeakyReLU_0.001_sl15_000_epoch_769.pt',
                   'test_data_path': 'dataset/v3_test_convert', 'scaler_path': 'data/MinMaxScaler_saved.pkl'}

data_configure = {'tx_power': 5, 'tx_height': 2, 'rx_height': 0.01, 'tx_antenna_gain': -1.47,
               'rx_antenna_gain': -1, 'environment': 1, 'device_id': 'f8:8a:5e:2d:80:f4',
               'use_fspl': True, 'save_dir': 'dataset/positioning_v1', 'scaler': 'None'}


def setup_test_data(model_configure, data_configure):
    path = model_configure['test_data_path']
    preprocessing.get_addition_dataset(path, data_configure)


def data_load(model_configure, data_configure, scaler):
    file_path = data_configure['save_dir']+'/dataset_0.csv'

    data = pd.read_csv(file_path, header=None)
    data = data.drop([data.columns[0]], axis=1)
    data = data.drop([data.columns[0]], axis=1)
    data = data.to_numpy()
    data = np.expand_dims(data, axis=0)

    return torch.tensor(data, dtype=torch.float)


def inference(model_configure, file):
    nn_model = model.model_load(model_configure)
    # 'dataset/v3_test_convert/dataset_2.csv'
    inference_dataloader = data_loader.load_path_loss_with_detail_inference_dataset(file, 'CRNN',
                                                                                    batch_size=model_configure['batch_size'])

    total_rssi = []
    total_label = []
    total_pred = []
    with torch.no_grad():
        for i, data in enumerate(inference_dataloader):
            if model_configure['model'] == 'DNN':
                pass
            elif model_configure['model'] == 'RNN':
                pass
            elif model_configure['model'] == 'CRNN':
                x_data = data.transpose(1, 2)
                x_data = x_data.cuda()
            y_pred = nn_model(x_data).reshape(-1).cpu().numpy()
            print(y_pred)
            print('mean : ', y_pred.mean())
            print('median : ', np.median(y_pred))
            print('max : ', y_pred.max())
            print('min : ', y_pred.min())
    return y_pred.max()


if __name__ == '__main__':
    # setup_test_data(model_configure, data_configure)
    inference(model_configure, 'dataset/v3_test_convert/dataset_2.csv')