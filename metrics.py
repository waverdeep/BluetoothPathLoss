import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from frechetdist import frdist


def mean_absolute_percentage_error(y_data, pred):
    y_data = np.array(y_data)
    pred = np.array(pred)
    return np.mean(np.abs((y_data-pred)/y_data))*100


def mean_percentage_error(y_data, pred):
    y_data = np.array(y_data)
    pred = np.array(pred)
    return np.mean((y_data-pred)/y_data)*100


def max_error_distance(y_data, pred):
    y_data = np.array(y_data)
    pred = np.array(pred)
    substract = np.abs(y_data - pred)
    return np.max(substract)


def min_error_distance(y_data, pred):
    y_data = np.array(y_data)
    pred = np.array(pred)
    substract = np.abs(y_data - pred)
    return np.min(substract)


def frechet_distance(rssi, y_data, pred):
    data_length = len(rssi)
    reshape_y_data = []
    reshape_pred = []
    for idx in range(data_length):
        reshape_y_data.append([rssi[idx], y_data[idx]])
        reshape_pred.append([rssi[idx], pred[idx]])

    frechet_result = frdist(reshape_y_data, reshape_pred)
    return frechet_result



if __name__ == '__main__':
    print('metrics test')
    # frechet distance
    # rssi, distance 쌍을 집어넣어야 확인할 수 있음
    P = [[1, 1], [2, 1]]
    Q = [[2, 2], [2, 1]]
    print('frechet distance : {}'.format(frdist(P, Q)))