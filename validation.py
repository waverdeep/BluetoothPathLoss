import torch
from data import data_loader
import model
from tool import metrics
import time
# randomness
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
# 연산속도가 느려질 수 있음
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU


def test(model_config, nn_model, dataloader, device):
    with torch.no_grad():
        total_rssi = []
        total_label = []
        total_pred = []
        for i, data in enumerate(dataloader):
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
            y_pred = nn_model(x_data).reshape(-1).cpu().numpy()
            x_data = x_data.cpu().numpy()
            y_data = y_data.cpu().numpy()
            rssi = x_data[:, 0]
            new_rssi = []
            for dd in rssi:
                new_rssi.append(dd[0])
            total_rssi += new_rssi
            total_label += y_data.tolist()
            total_pred += y_pred.tolist()

            print(len(new_rssi))
            print(len(y_data))
            print(len(y_pred))

            metrics.plot_rssi_to_distance(new_rssi, y_data, y_pred)
            print('<<iter metric result>>')
            metrics.show_all_metrics_result(new_rssi, y_data.tolist(), y_pred.tolist())

        print('<<total metric result>>')
        print(total_rssi)
        print(total_label)
        print(total_pred)
        metrics.show_all_metrics_result(total_rssi, total_label, total_pred)


def setting(model_config, checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, test_dataloader, _ = data_loader.load_path_loss_with_detail_dataset(input_dir=model_config['input_dir'],
                                                                           model_type=model_config['model'],
                                                                           num_workers=model_config['num_workers'],
                                                                           batch_size=model_config['batch_size'],
                                                                           shuffle=model_config['shuffle'],
                                                                           input_size=model_config['input_size'] if
                                                                           model_config['model'] == 'DNN' else model_config['sequence_length'])
    nn_model = model.model_load(model_config)

    checkpoint = torch.load(checkpoint_path)
    nn_model.load_state_dict(checkpoint['model_state_dict'])
    test(model_config=model_config, nn_model=nn_model, dataloader=test_dataloader, device=device)


if __name__ == '__main__':

    checkpoint_path = 'checkpoints_all/CRNN_Adam_LeakyReLU_0.001_sl15_010_epoch_729.pt'
    configure = {"model": "CRNN",
                 "criterion": "MSELoss",
                 "optimizer": "Adam","activation": "LeakyReLU",
                 "learning_rate": 0.001,
                 "cuda": True,
                 "batch_size": 512,
                 "epoch": 1000,
                 "input_size": 8,
                 "sequence_length": 15,
                 "input_dir": "dataset/v1_scaled",
                 "shuffle": True,
                 "num_workers": 8}
    start_time = time.time()
    setting(configure, checkpoint_path)
    print(time.time() - start_time)
