import numpy as np

def MAPE(y_data, pred):
    y_data = np.array(y_data)
    pred = np.array(pred)

    return np.mean(np.abs((y_data-pred)/y_data))*100


def MPE(y_data, pred):
    y_data = np.array(y_data)
    pred = np.array(pred)

    return np.mean((y_data-pred)/y_data)*100
