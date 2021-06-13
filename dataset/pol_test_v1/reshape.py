import glob
import os
import pandas as pd


def get_all_file_path(input_dir, file_extension):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp

filelist = get_all_file_path('./', 'csv')

for file in filelist:
    data = pd.read_csv(file, names=['rssi', 'tx_power', 'tx_height', 'rx_height', 'tx_antenna_gain','rx_antenna_gain','FSPL', 'environment'])
    data = data[['rssi', 'tx_power', 'tx_height', 'rx_height', 'tx_antenna_gain','rx_antenna_gain','environment','FSPL']]
    data.to_csv(file, header=False, index=False)