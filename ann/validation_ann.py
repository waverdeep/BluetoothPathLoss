import torch
from data import data_loader
import model
import tools

# parameters
input_size = 8
batch_size = 64
checkpoint_path = 'checkpoints/model_adadelta_mse_epoch_529.pt'
valid_data_dir = 'dataset/pathloss_v1_valid_cs.csv'

# gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dataloader
valid_dataloader = data_loader.load_pathloss_dataset(valid_data_dir, batch_size, True, 4)

# model load
model = model.VanillaNetwork(input_size).cuda()
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

with torch.no_grad():
    total_rssi = []
    total_label = []
    total_pred = []
    for i, data in enumerate(valid_dataloader):
        x_data, y_data = data
        x_data = x_data.cuda()
        y_data = y_data.cuda()
        pred = model(x_data)
        pred = pred.squeeze(-1)
        rssi = x_data[:, 0]
        rssi = rssi.cpu().numpy()
        y_data = y_data.cpu().numpy()
        pred = pred.cpu().numpy()

        total_rssi += rssi.tolist()
        total_label += y_data.tolist()
        total_pred += pred.tolist()

        tools.plot_rssi_to_distance(rssi, y_data, pred)
        print('<<iter metric result>>')
        tools.show_all_metrics_result(rssi.tolist(), y_data.tolist(), pred.tolist())

    print('<<total metric result>>')
    tools.show_all_metrics_result(total_rssi, total_label, total_pred)