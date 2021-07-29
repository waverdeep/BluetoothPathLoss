import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from tool import file_io


def cast_float_tensor(data):
    return torch.tensor(data, dtype=torch.float)


def data_split(dataset, test_size=0.3):
    train_data, test_data = train_test_split(dataset, test_size=test_size, shuffle=True, random_state=42)
    test_data, valid_data = train_test_split(test_data, test_size=0.5, shuffle=True, random_state=42)
    return train_data, test_data, valid_data


class PathLossWithDetailDataset(Dataset):
    def __init__(self, input_data,  model_type='DNN'):
        self.model_type = model_type
        self.dataset = input_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pick = self.dataset[idx]
        if self.model_type == 'DNN' or self.model_type == 'Custom_DNN':
            y_label = pick[0]
            x_data = pick[1:]
            return cast_float_tensor(x_data), cast_float_tensor(y_label)
        elif self.model_type == 'RNN' or self.model_type == 'CRNN' or self.model_type == 'Custom_RNN' or self.model_type == 'Custom_CRNN':
            y_label = pick[0][0]
            x_data = np.delete(pick, 0, axis=1)
            return cast_float_tensor(x_data), cast_float_tensor(y_label)


def load_path_loss_with_detail_dataset(input_dir, model_type='RNN', num_workers=4, batch_size=128,
                                       shuffle=True, input_size=10):
    # 파일들이 저장되었는 경로를 받아 파일 리스트를 얻어냄
    file_list = file_io.get_all_file_path(input_dir, file_extension='csv')
    # csv에 있는 모든 데이터를 다 꺼내서 넘파이로 만듬
    addition_dataset = []
    setup_dataset = None
    for idx, file in enumerate(file_list):
        addition_dataset.append(pd.read_csv(file).to_numpy())
    if model_type == 'DNN' or model_type == 'Custom_DNN':
        dnn_dataset = []
        for pack in addition_dataset:
            for line in pack:
                dnn_dataset.append(line)
        setup_dataset = dnn_dataset
    elif model_type == 'RNN' or model_type == 'CRNN' or model_type == 'Custom_RNN' or model_type == 'Custom_CRNN':
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
        setup_dataset = rnn_dataset

    train_data, test_data, valid_data = data_split(setup_dataset)
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


class PathLossWithDetailInferenceDataset(Dataset):
    def __init__(self, input_data):
        self.dataset = input_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pick = self.dataset[idx]
        return cast_float_tensor(pick)


def load_path_loss_with_detail_inference_dataset(input_filepath, model_type='RNN', num_workers=4, batch_size=128,
                                                 shuffle=True, input_size=10):
    addition_dataset = []
    setup_dataset = None
    addition_dataset.append(pd.read_csv(input_filepath).to_numpy())
    if model_type == 'DNN':
        dnn_dataset = []
        for pack in addition_dataset:
            for line in pack:
                dnn_dataset.append(line)
        setup_dataset = dnn_dataset
    elif model_type == 'RNN' or model_type == 'CRNN':
        rnn_dataset = []
        for pack in addition_dataset:
            for idx in range((len(pack) - input_size)):
                rnn_dataset.append(pack[idx: idx+input_size])
        rnn_dataset = np.array(rnn_dataset)
        setup_dataset = rnn_dataset
    inference_dataset = PathLossWithDetailInferenceDataset(input_data=setup_dataset)
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return inference_dataloader


if __name__ == '__main__':
    train_dataloader, test_dataloader, validation_dataloader = load_path_loss_with_detail_dataset(
        '../dataset/v5/new', model_type='Custom_CRNN')
    print('train_dataloader : ', len(train_dataloader))
    for data in train_dataloader:
        print(data[:][0].shape)
        print(data[:][1].shape)
        break

    # inference_dataloader = load_path_loss_with_detail_inference_dataset('../dataset/v1/pol_test_v1/POL_A_cs_6_20.csv')
    # for data in inference_dataloader:
    #     print(data.shape)
    #     break



