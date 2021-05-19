import torch
from data import data_loader
import model
from tool import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.externals import joblib
torch.manual_seed(42)


def test(model_config, nn_model, dataloader, device):
    scaler_name = "./data/MinMaxScaler_saved.pkl"
    scaler = joblib.load(scaler_name)
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
            y_pred = nn_model(x_data).reshape(-1).cpu().numpy()
            x_data = x_data.cpu().numpy()
            y_data = y_data.cpu().numpy()
            rssi = x_data[:, 0]
            new_rssi = []
            for line in rssi:
                new_rssi.append(line.tolist())
            first_rssi = []
            for dd in new_rssi:
                dd = [dd]
                temp = scaler.inverse_transform(dd).tolist()
                first_rssi.append(temp[0][0])
            print(first_rssi)
            total_rssi += first_rssi
            total_label += y_data.tolist()
            total_pred += y_pred.tolist()


            metrics.plot_rssi_to_distance(first_rssi, y_data, y_pred)
            print('<<iter metric result>>')
            metrics.show_all_metrics_result(first_rssi, y_data.tolist(), y_pred.tolist())

        print('<<total metric result>>')
        metrics.show_all_metrics_result(total_rssi, total_label, total_pred)


def setting(model_config, checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nn_model = None
    _, test_dataloader, _ = data_loader.load_path_loss_with_detail_dataset(input_dir=model_config['input_dir'],
                                                                           model_type=model_config['model'],
                                                                           num_workers=model_config['num_workers'],
                                                                           batch_size=model_config['batch_size'],
                                                                           shuffle=model_config['shuffle'],
                                                                           input_size=model_config['input_size'] if
                                                                           model_config['model'] == 'FFNN' else model_config['sequence_length'])
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

    checkpoint = torch.load(checkpoint_path)
    nn_model.load_state_dict(checkpoint['model_state_dict'])
    test(model_config=model_config, nn_model=nn_model, dataloader=test_dataloader, device=device)

if __name__ == '__main__':
    checkpoint_path = 'checkpoints_v2/testcase_configuration_cnnrnn_adamw_001_epoch_1449.pt'
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
    setting(configure, checkpoint_path)
