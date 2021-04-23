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


if __name__ == '__main__':
    print('metrics test')
