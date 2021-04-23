import matplotlib.pyplot as plt
import metrics
import numpy as np


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
    mse_result = metrics.mean_squared_error(y_data, pred)
    rmse_result = np.sqrt(mse_result)
    r2_result = metrics.r2_score(y_data, pred)
    mae_result = metrics.mean_absolute_error(y_data, pred)
    max_error_result = metrics.max_error_distance(y_data, pred)
    min_error_result = metrics.min_error_distance(y_data, pred)
    frechet_result = metrics.frechet_distance(rssi, y_data, pred)

    print('MSE : {}'.format(mse_result))
    print('RMSE : {}'.format(rmse_result))
    print('R2 Score : {}'.format(r2_result))
    print('MAE : {}'.format(mae_result))
    print('MAX_ERROR_DISTANCE : {}'.format(max_error_result))
    print('MIN_ERROR_DISTANCE : {}'.format(min_error_result))
    print('FRECHET_DISTANCE : {}'.format(frechet_result))



