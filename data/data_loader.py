import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


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
    def __init__(self, input_dir, sequence_length): # 리스트로 받을 지 결정
        self.input_dir = input_dir
        self.dataset = self.reshape_dataset(sequence_length)
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        pass

    def reshape_dataset(self, sequence_length):
        dataset = []
        divide_dataset = []
        files = []
        meter_status = -1
        for file in self.input_dir:
            raw_data = pd.read_csv(file, names=['mac', 'meter', 'rssi',
                                                'tx_power', 'tx_height',
                                                'rx_height', 'tx_antenna_gain',
                                                'rx_antenna_gain', 'FSPL',
                                                'environment'])
            temp = []
            for index, row in raw_data.iterrows():
                if row['meter'] != meter_status:
                    if len(temp) != 0:
                        divide_dataset.append(temp)
                        temp = []
                        meter_status = row['meter']
                        temp.append(row.tolist())

                temp.append(row.tolist())

        # ----- #
        for div in divide_dataset:
            div_len = len(div)
            for j in range(0, (div_len-sequence_length)):
                dataset.append(div[j:j+sequence_length])



        return dataset


def load_pathloss_dataset(input_dir, batch_size, shuffle, num_workers, type='ANN'):
    if type == 'ANN':
        pathloss_dataset = PathLossDataset(input_dir=input_dir, scaler='robust')
        pathloss_dataloader = DataLoader(pathloss_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return pathloss_dataloader
    elif type == 'RNN':
        pass
