import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import glob
import os
from tool import path_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.externals import joblib


def set_scaler(name):
    if name == 'MinMaxScaler':
        return MinMaxScaler()
    elif name == 'RobustScaler':
        return RobustScaler()
    elif name == 'StandardScaler':
        return StandardScaler()


def get_all_file_path(input_dir, file_extension):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp


class PathLossWithDetailDataset(Dataset):
    def __init__(self, input_dir):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass



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


def fit_selected_scaler(addition_dataset, scaler='None'):
    tied_scaler = None
    if scaler != 'None' and scaler == 'MinMaxScaler':
        temp = []
        tied_scaler = set_scaler(scaler)
        for pack in addition_dataset:
            for line in pack:
                temp.append(line)
        temp = np.array(temp)
        temp = np.delete(temp, 0, axis=1)
        tied_scaler.fit(temp)
    save_scaler_name = './{}_saved.pkl'.format(scaler)
    joblib.dump(tied_scaler, save_scaler_name)
    return tied_scaler


def load_path_loss_with_detail_dataset(input_dir, device_id, data_environment, model_type='RNN', scaler='None', use_fspl=False, use_r2d=False):
    # 파일들이 저장되었는 경로를 받아 파일 리스트를 얻어냄
    file_list = get_all_file_path(input_dir, file_extension='csv')
    target_dataset = []
    addition_dataset = []

    for file in file_list:
        temp = pd.read_csv(file)
        temp = temp[temp['mac'] == device_id]
        temp = temp.drop(['mac', 'type'], axis=1)
        target_dataset.append(temp)

    for item in target_dataset:
        temp = []
        # meter, rssi, tx_power, tx_height, rx_height, tx_antenna_gain, rx_antenna_gain, environment, FSPL,
        for idx, line in item.iterrows():
            data = line.tolist()
            data.append(data_environment.get('tx_power'))
            data.append(data_environment.get('tx_height'))
            data.append(data_environment.get('rx_height'))
            data.append(data_environment.get('tx_antenna_gain'))
            data.append(data_environment.get('rx_antenna_gain'))
            data.append(data_environment.get('environment'))
            if use_fspl:
                data.append(path_loss.get_distance_with_rssi_fspl(data[1]))
            temp.append(data)
        addition_dataset.append(temp)

    tied_scaler = fit_selected_scaler(addition_dataset=addition_dataset, scaler=scaler)










# board : f8:8a:5e:2d:80:f4
# custom : 04:ee:03:74:ae:dd
# environment
# - grass : 1
# - block : 2
# weather 에 대한 상태가 들어가면 좋을지를 고민해 보아야 한다. 혹은 기온이라더지 등
if __name__ == '__main__':
    data_environment = {'tx_power': 5, 'tx_height': 2, 'rx_height': 0.01, 'tx_antenna_gain': -1.47,
                        'rx_antenna_gain': -1, 'environment': 1}
    load_path_loss_with_detail_dataset('../dataset/v1_timeline', device_id='f8:8a:5e:2d:80:f4', data_environment=data_environment,
                                       use_fspl=True, scaler='MinMaxScaler', model_type='RNN')

