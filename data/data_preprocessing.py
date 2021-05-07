import pandas as pd
from sklearn.model_selection import train_test_split
import tool.path_loss as path_loss
import glob
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


def get_all_file_path(input_dir, file_extension):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp


def get_filename(input_filepath):
    return input_filepath.split('/')[-1]


def get_pure_filename(input_filepath):
    temp = input_filepath.split('/')[-1]
    return temp.split('.')[0]


def create_directory(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except OSError:
        print('Error : Creating directory: '+dir_path)


def set_scaler(name):
    if name == 'MinMaxScaler':
        return MinMaxScaler()
    elif name == 'RobustScaler':
        return RobustScaler()
    elif name == 'StandardScaler':
        return StandardScaler()


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


def get_train_test_valid():
    data = pd.read_csv('../dataset/v1_timeline')
    data2 = pd.read_csv('dataset/2021_04_15/rssi_s.csv')

    result = pd.concat([data, data2])
    total = []
    for index, row in result.iterrows():
        row = row.tolist()
        meter = row[0]
        rssi = row[3]
        mac = row[1]
        # mac, meter, rssi, tx_power, tx_height, rx_height, tx_antenna_gain, rx_antenna_gain, FSPL, environment
        total.append([mac, meter, rssi, 5, 0.1, 2, -1.47, -1, path_loss.get_distance_with_rssi_fspl(rssi), 1])

    df = pd.DataFrame(total)
    df.to_csv('dataset/2021_04_15/pathloss_v1.csv', header=None, index=None)

    train, test = train_test_split(df, test_size=0.3, shuffle=True, random_state=32)
    test, valid = train_test_split(test, test_size=0.2, shuffle=True, random_state=32)

    train.to_csv('dataset/2021_04_15/pathloss_v1_train.csv', header=None, index=None)
    test.to_csv('dataset/2021_04_15/pathloss_v1_test.csv', header=None, index=None)
    valid.to_csv('dataset/2021_04_15/pathloss_v1_valid.csv', header=None, index=None)


def get_addition_dataset(input_dir, config):
    file_list = get_all_file_path(input_dir=input_dir, file_extension='csv')
    target_dataset = []
    addition_dataset = []

    for file in file_list:
        temp = pd.read_csv(file)
        temp = temp[temp['mac'] == config['device_id']]
        temp = temp.drop(['mac', 'type'], axis=1)
        target_dataset.append(temp)

    for item in target_dataset:
        temp = []
        for idx, line in item.iterrows():
            data = line.tolist()
            data.append(config.get('tx_power'))
            data.append(config.get('tx_height'))
            data.append(config.get('rx_height'))
            data.append(config.get('tx_antenna_gain'))
            data.append(config.get('rx_antenna_gain'))
            data.append(config.get('environment'))
            if config['use_fspl']:
                data.append(path_loss.get_distance_with_rssi_fspl(data[1]))
            temp.append(data)
        addition_dataset.append(temp)

    if config['scaler'] == 'None':
        pass
    if config['scaler'] == 'MinMaxScaler':
        tied_scaler = fit_selected_scaler(addition_dataset=addition_dataset, scaler=config['scaler'])
        for idx, item in enumerate(addition_dataset):
            temp = np.array(item)
            y_label = temp[:, 0]
            y_label = np.expand_dims(y_label, axis=1)
            print(y_label)
            x_data = np.delete(temp, 0, axis=1)
            x_data = tied_scaler.transform(x_data)
            total = np.concatenate((y_label, x_data), axis=1)
            addition_dataset[idx] = total

    for idx, item in enumerate(addition_dataset):
        temp = pd.DataFrame(item)
        create_directory(config['save_dir'])
        temp.to_csv('{}/dataset_{}.csv'.format(config['save_dir'], idx) , header=None, index=None)


# 04:ee:03:74:ae:dd -> 골프공
# f8:8a:5e:2d:80:f4 -> R1
if __name__ == '__main__':
    input_dir = '../dataset/v2_timeline'
    device_id = 'f8:8a:5e:2d:80:f4'
    config = {
        'tx_power': 5, 'tx_height': 2, 'rx_height': 0.01, 'tx_antenna_gain': -1.47,
        'rx_antenna_gain': -1, 'environment': 1, 'device_id': device_id,
        'use_fspl': True, 'save_dir': '../dataset/v1_scaled/', 'scaler': 'None'
    }
    get_addition_dataset(input_dir=input_dir, config=config)

