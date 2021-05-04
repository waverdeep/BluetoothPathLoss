import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import glob
import os
from sklearn.model_selection import train_test_split
import numpy as np


def get_all_file_path(input_dir, file_extension):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp


def data_split(dataset, test_size=0.3):
    train_data, test_data = train_test_split(dataset, test_size=test_size, shuffle=True, random_state=42)
    test_data, valid_data = train_test_split(test_data, test_size=test_size, shuffle=True, random_state=42)
    return train_data, test_data, valid_data

class PathLossWithDetailDataset(Dataset):
    def __init__(self, input_data,  model_type='FFNN'):
        self.model_type = model_type
        self.dataset = input_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pick = self.dataset[idx]
        if self.model_type == 'FFNN':
            y_label = pick[0]
            x_data = pick[1:]
            return torch.tensor(x_data, dtype=torch.float), torch.tensor(y_label, dtype=torch.float)
        elif self.model_type == 'RNN':
            y_label = pick[0][0]
            x_data = np.delete(pick, 0, axis=1)
            return torch.tensor(x_data, dtype=torch.float), torch.tensor(y_label, dtype=torch.float)

def load_path_loss_with_detail_dataset(input_dir, model_type='RNN',
                                       num_workers=4, batch_size=62, shuffle=True, input_size=10):
    # 파일들이 저장되었는 경로를 받아 파일 리스트를 얻어냄
    file_list = get_all_file_path(input_dir, file_extension='csv')
    addition_dataset = []
    for idx, file in enumerate(file_list):
        addition_dataset.append(pd.read_csv(file).to_numpy())

    if model_type == 'FFNN':
        ffnn_dataset = []
        for pack in addition_dataset:
            for line in pack:
                ffnn_dataset.append(line)
        train_data, test_data, valid_data = data_split(ffnn_dataset)
        pathloss_train_dataset = PathLossWithDetailDataset(input_data=train_data,
                                                           model_type=model_type)
        pathloss_test_dataset = PathLossWithDetailDataset(input_data=test_data,
                                                          model_type=model_type)
        pathloss_valid_dataset = PathLossWithDetailDataset(input_data=valid_data,
                                                           model_type=model_type)
        pathloss_train_dataloader = DataLoader(pathloss_train_dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers)
        pathloss_test_dataloader = DataLoader(pathloss_test_dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers)
        pathloss_valid_dataloader = DataLoader(pathloss_valid_dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers)
        return pathloss_train_dataloader, pathloss_test_dataloader, pathloss_valid_dataloader

    elif model_type == 'RNN':
        div_meter_pack = []
        rnn_dataset = []
        for n_idx, pack in enumerate(addition_dataset):
            label = pack[:, 0].tolist()
            label = list(set(label))
            temp_pack = pd.DataFrame(pack)
            for key in label:
                div_meter_pack.append(temp_pack[temp_pack[0] == key].to_numpy())

        for n_idx, pack in enumerate(div_meter_pack):
            for i in range(len(pack)-input_size):
                rnn_dataset.append(pack[i:i+input_size])

        rnn_dataset = np.array(rnn_dataset)

        train_data, test_data, valid_data = data_split(rnn_dataset)
        pathloss_train_dataset = PathLossWithDetailDataset(input_data=train_data,
                                                           model_type=model_type)
        pathloss_test_dataset = PathLossWithDetailDataset(input_data=test_data,
                                                          model_type=model_type)
        pathloss_valid_dataset = PathLossWithDetailDataset(input_data=valid_data,
                                                           model_type=model_type)
        pathloss_train_dataloader = DataLoader(pathloss_train_dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers)
        pathloss_test_dataloader = DataLoader(pathloss_test_dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers)
        pathloss_valid_dataloader = DataLoader(pathloss_valid_dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers)
        return pathloss_train_dataloader, pathloss_test_dataloader, pathloss_valid_dataloader


if __name__ == '__main__':
    load_path_loss_with_detail_dataset('../dataset/v1_scaled', model_type='RNN')

























##-------------------old version code-----------------------##
class PathLossDataset(Dataset):
    def __init__(self, input_dir, scaler='None'):
        self.dataset = pd.read_csv(input_dir, header=None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        line = self.dataset.loc[idx]
        line = line.tolist()
        del line[0]
        y_data = line[0]
        del line[0]

        x_data = line
        x_data = torch.tensor(x_data, dtype=torch.float)
        y_data = torch.tensor(y_data, dtype=torch.float)
        return x_data, y_data


class PathLossTimeSeriesDataset(Dataset):
    # dataset
    # --> train관련 파일이 1개 혹은 2개 3개가 있을 수 있음
    # 정렬이 제대로 되어있다고 가정하고 코드를 짜야 함
    # 즉 전처리 과정에서 데이터가 꼭 잘 들어있어야 함. (추후에는 데이터셋에 TimeStamp를 추가해야할 것)
    def __init__(self, dataset, scaler='None'): # 리스트로 받을 지 결정
        self.dataset = dataset
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        file = self.dataset[idx]
        file = pd.read_csv(file, header=None)
        y_label = file[0][0]
        # y_label = y_label.to_numpy()
        file = file.drop(file.columns[0], axis='columns')
        x_label = file.to_numpy()
        return torch.tensor(x_label, dtype=torch.float), torch.tensor(y_label, dtype=torch.float)


def load_pathloss_dataset(input_dir, batch_size, shuffle, num_workers, type='ANN'):
    if type == 'ANN':
        pathloss_dataset = PathLossDataset(input_dir=input_dir, scaler='robust')
        pathloss_dataloader = DataLoader(pathloss_dataset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers)
        return pathloss_dataloader
    elif type == 'RNN':
        pathloss_dataset = PathLossTimeSeriesDataset(dataset=input_dir, scaler='robust')
        pathloss_dataloader = DataLoader(pathloss_dataset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers)
        return pathloss_dataloader