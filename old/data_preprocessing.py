import pandas as pd
from sklearn.model_selection import train_test_split
import model.model_pathloss as path_loss
import tool.file_io as file_io


def grapped_raw_data():
    pass


def get_addition_dataset(config):
    file_list = file_io.get_all_file_path(input_dir=config['input_dir'], file_extension='csv')
    file_list.sort()
    print(file_list)
    target_dataset = []
    addition_dataset = []

    # original file have -> ['meter', 'mac', 'type', 'rssi']
    # version up file have -> ['meter', 'mac', 'type', 'rssi', 'channel']
    for file in file_list:
        temp = pd.read_csv(file)
        if config['device_id'] != '':
            temp = temp[temp['mac'] == config['device_id']]
        temp = temp.drop(['mac', 'type'], axis=1) # 불필요한 데이터 제거
        target_dataset.append(temp)
    # dropped -> ['meter', 'rssi', 'channel']

    for item in target_dataset:
        temp = []
        for idx, line in item.iterrows():
            data = line.tolist()
            data.append(config.get('tx_power'))
            data.append(config.get('rx_height'))
            data.append(config.get('tx_antenna_gain'))
            data.append(config.get('rx_antenna_gain'))
            data.append(config.get('covered'))
            if config['use_fspl']:
                data.append(path_loss.get_distance_with_rssi_fspl(data[1]))
            if config['inference']:
                del data[0]
            temp.append(data)
        addition_dataset.append(temp)
    # finish ['meter', 'rssi', 'channel', 'tx_power', 'rx_height', 'tx_antenna_gain',
    #         'rx_antenna_gain', 'covered', 'fspl']

    for idx, item in enumerate(addition_dataset):
        temp = pd.DataFrame(item)
        file_io.create_directory(config['save_dir'])
        temp.to_csv('{}/dataset_{}_mac_{}.csv'.format(config['save_dir'], idx, config['device_id']), header=None, index=None)


if __name__ == '__main__':
    config_set = {
        'input_dir': '../dataset/v4_test/dk_5_5',
        'save_dir': '../dataset/v4_test/dk_convert_5_5/',
        'tx_power': 8,
        'rx_height': 2.0,
        'tx_antenna_gain': -1,
        'rx_antenna_gain': -1,
        'covered': 1,
        'device_id': 'f8:8a:5e:2d:80:f4', # optional
        'use_fspl': True,
        'inference': False
    }
    get_addition_dataset(config=config_set)


#### not used #############################################################################################
def get_train_test_valid():
    data = pd.read_csv('../dataset/v1/v1_timeline')
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

