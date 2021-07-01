import math
from data import data_loader
import numpy as np
from tool import metrics


def egli_model(rssi, hm=2, hb=0.01, f=2400):
    result = pow(10, (-rssi-20*math.log10(f) + 20*math.log10(hb)+10*math.log10(hm)-76.3)/40)
    return result


def trgr_model(rssi, gt=1.47, gr=1, ht=2, hr=0.01):
    result = pow(10, (-rssi + 5+10*math.log10(gt*gr)+20*math.log10(ht*hr))/40)
    return result


def fspl_model(rssi, freq=2400):
    result = math.pow(10, (-rssi - 20 * math.log10(freq) + 27.55) / 20)
    return result


def fspl_model_inverse(distance, freq=2400):
    result = 20 * math.log10(distance) + 20 * math.log10(freq) - 27.55
    return -result


def get_distance_with_rssi(rssi, tx_power, free_space=2):
    return math.pow(10, (tx_power - rssi) / (10 * free_space))


def log_distance(rssi):
    freq = 2412 * math.pow(10, 6)  # Transmission frequency
    txPower = 5  # Transmission power in dB
    antennaeGain = 0  # Total antennae gains (transmitter + receiver) in dB
    refDist = 1  # Reference distance from the transceiver in meters
    lightVel = 3e8  # Speed of light
    refLoss = 20 * np.log10(
        (4 * np.pi * refDist * freq / lightVel))  # Free space path loss at the reference distance (refDist)
    ERP = txPower + antennaeGain - refLoss  # Kind of the an equivalent radiation power
    return math.pow(10, ((rssi - ERP + 17.76) / (-10 * 1.92)))


def test(dataloader, model_type=''):
    total_rssi = []
    total_label = []
    total_pred = []
    for i, data in enumerate(dataloader):
        x_data, y_data = data
        x_data = np.array(x_data)
        for i in range (len(x_data)):
            if float(y_data[i]) == 0:
                continue
            x = x_data[i][0]
            print(x)
            pred = 0
            if model_type == 'egli':
                pred = egli_model(x)*1000
            elif model_type == 'trgr':
                pred = trgr_model(x)
            elif model_type == 'fspl':
                pred = fspl_model(x)
            total_pred.append(pred)
            total_label.append(float(y_data[i]))
            total_rssi.append(x)


    print(total_label)
    print(total_pred)
    metrics.show_all_metrics_result(total_rssi, total_label, total_pred)
    metrics.plot_rssi_to_distance(total_rssi, total_label, total_pred)









def main(model_config):
    _, test_dataloader, _ = data_loader.load_path_loss_with_detail_dataset(input_dir=model_config['input_dir'],
                                                                           model_type=model_config['model'],
                                                                           num_workers=model_config['num_workers'],
                                                                           batch_size=model_config['batch_size'],
                                                                           shuffle=model_config['shuffle'],
                                                                           input_size=model_config['input_size'] if
                                                                           model_config['model'] == 'DNN' else
                                                                           model_config['sequence_length'])
    test(dataloader=test_dataloader, model_type='fspl')



if __name__ == '__main__':
    configure = {"model": "DNN",
                 "criterion": "MSELoss",
                 "optimizer": "Adam", "activation": "LeakyReLU",
                 "learning_rate": 0.001,
                 "cuda": True,
                 "batch_size": 512,
                 "epoch": 1000,
                 "input_size": 8,
                 "sequence_length": 15,
                 "input_dir": "dataset/v1_scaled",
                 "shuffle": True,
                 "num_workers": 8}
    main(configure)