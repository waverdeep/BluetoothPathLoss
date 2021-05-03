import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from frechetdist import frdist
import matplotlib.pyplot as plt


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


def plot_rssi_to_distance(ref_rssi, y_data, pred):
    plt.figure(figsize=(10, 10))
    plt.title('Validation RSSI to Distance', fontsize=20)
    plt.ylabel('Distance (meter)', fontsize=18)
    plt.xlabel('RSSI (dBm)', fontsize=18)
    plt.scatter(ref_rssi, y_data, c='b', label="groundtruth")
    plt.scatter(ref_rssi, pred, c='r', label="prediction")
    plt.legend()
    plt.grid(True)
    plt.show()


def show_all_metrics_result(rssi, y_data, pred):
    mse_result = mean_squared_error(y_data, pred)
    rmse_result = np.sqrt(mse_result)
    r2_result = r2_score(y_data, pred)
    mae_result = mean_absolute_error(y_data, pred)
    max_error_result = max_error_distance(y_data, pred)
    min_error_result = min_error_distance(y_data, pred)
    frechet_result = frechet_distance(rssi, y_data, pred)

    print('MSE : {}'.format(mse_result))
    print('RMSE : {}'.format(rmse_result))
    print('R2 Score : {}'.format(r2_result))
    print('MAE : {}'.format(mae_result))
    print('MAX_ERROR_DISTANCE : {}'.format(max_error_result))
    print('MIN_ERROR_DISTANCE : {}'.format(min_error_result))
    print('FRECHET_DISTANCE : {}'.format(frechet_result))


if __name__ == '__main__':
    print('metrics test')
    # frechet distance
    # rssi, distance 쌍을 집어넣어야 확인할 수 있음
    P = [[1, 1], [2, 1]]
    Q = [[2, 2], [2, 1]]
    print('frechet distance : {}'.format(frdist(P, Q)))