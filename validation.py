import torch
import data_loader
import model
import matplotlib.pyplot as plt


input_size = 8
batch_size = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint_path = 'checkpoints/model_adadelta_mse_epoch_400.pt'
valid_data_dir = 'dataset/pathloss_v1_valid_ev.csv'
valid_dataloader = data_loader.load_pathloss_dataset(valid_data_dir, batch_size, True, 4)

model = model.VanillaNetwork(input_size).cuda()

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)

with torch.no_grad():
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

        total_label += y_data.tolist()
        total_pred += pred.tolist()

        plt.scatter(y_data, rssi, c='b')
        plt.scatter(pred, rssi, c='y')
        plt.legend()
        plt.show()

