import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from tool import file_io
from sklearn.preprocessing import RobustScaler
from sklearn.externals import joblib


def cast_float_tensor(data):
    return torch.tensor(data, dtype=torch.float)


def data_split(dataset, test_size=0.3, shuffle=True):
    train_data, valid_data = train_test_split(dataset, test_size=test_size, shuffle=shuffle, random_state=42)
    test_data = None
    return train_data, valid_data, test_data


class PathLossWithDetailDataset(Dataset):
    def __init__(self, input_data,  model_type='CRNN', scaler_type='robust'):
        self.model_type = model_type
        self.dataset = input_data
        self.scaler_type = scaler_type
        self.scaler = joblib.load('./data/{}_scaler.pkl'.format(scaler_type))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pick = self.dataset[idx]
        y_label = pick[0][0]
        x_data = np.delete(pick, 0, axis=1)
        if self.scaler_type == 'robust':
            x_data = self.scaler.transform(x_data)
        return cast_float_tensor(x_data), cast_float_tensor(y_label)


def get_scaler_checkpoint(input_dir, scaler_type='robust'):
    file_list = file_io.get_all_file_path(input_dir, file_extension='csv')

    dataset = pd.DataFrame()

    for idx, file in enumerate(file_list):
        temp = pd.read_csv(file, header=None)
        dataset = pd.concat([dataset, temp], ignore_index=True)

    x_data = dataset.drop([0], axis='columns')

    if scaler_type == 'robust':
        scaler = RobustScaler().fit(x_data)
        filename = '{}_scaler.pkl'.format(scaler_type)
        joblib.dump(scaler, filename)


def load_path_loss_with_detail_dataset(input_dir, model_type="CRNN", num_workers=8, batch_size=128,
                                       shuffle=True, input_size=15, various_input=False, scaler_type='robust'):
    file_list = file_io.get_all_file_path(input_dir, file_extension='csv')
    addition_dataset = []

    for idx, file in enumerate(file_list):
        addition_dataset.append(pd.read_csv(file).to_numpy())

    div_meter_pack = []
    rnn_dataset = []
    for n_idx, pack in enumerate(addition_dataset):
        label = pack[:, 0].tolist()
        label = list(set(label))
        temp_pack = pd.DataFrame(pack)
        for key in label:
            div_meter_pack.append(temp_pack[temp_pack[0] == key].to_numpy())

    for n_idx, pack in enumerate(div_meter_pack):
        if len(pack) < 30:
            temp = pack.tolist()
            temp = temp * (int(30 / len(pack)) + 2)
            pack = np.array(temp)
        for i in range(len(pack)-input_size):
            rnn_dataset.append(pack[i:i+input_size])

    rnn_dataset = np.array(rnn_dataset)
    setup_dataset = rnn_dataset

    train_data, valid_data, test_data = data_split(setup_dataset, shuffle=shuffle)
    pathloss_train_dataset = PathLossWithDetailDataset(input_data=train_data,
                                                       model_type=model_type,
                                                       scaler_type=scaler_type)
    pathloss_test_dataset = None
    pathloss_valid_dataset = PathLossWithDetailDataset(input_data=valid_data,
                                                       model_type=model_type,
                                                       scaler_type=scaler_type)
    pathloss_train_dataloader = DataLoader(pathloss_train_dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers)
    pathloss_test_dataloader = None
    pathloss_valid_dataloader = DataLoader(pathloss_valid_dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers)
    return pathloss_train_dataloader, pathloss_valid_dataloader, pathloss_test_dataloader

if __name__ == '__main__':
    get_scaler_checkpoint("../dataset/v9/type01_train")
    train_dataloader, test_dataloader, validation_dataloader = load_path_loss_with_detail_dataset(
        '../dataset/v8/type01_train', model_type='Custom_CRNN', batch_size=4)
    print('train_dataloader : ', len(train_dataloader))
    for data in train_dataloader:
        for temp in data[:][0]:
            print("dd", temp[:, 0])
        print(data[:][1].shape)
        break