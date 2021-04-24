import math
import pandas as pd
from sklearn.model_selection import train_test_split


def get_distance_with_rssi_fspl(rssi, freq=2412 * math.pow(10, 6)):
    return math.pow(10, (-rssi - 20 * math.log10(freq) + 147.55) / 20)


data = pd.read_csv('dataset/2021_04_15/rssi.csv')
data2 = pd.read_csv('dataset/2021_04_15/rssi_s.csv')

result = pd.concat([data, data2])
total = []
for index, row in result.iterrows():
    row = row.tolist()
    meter = row[0]
    rssi = row[3]
    mac = row[1]
    # mac, meter, rssi, tx_power, tx_height, rx_height, tx_antenna_gain, rx_antenna_gain, FSPL, environment
    total.append([mac, meter, rssi, 5, 0.1, 2, -1.47, -1, get_distance_with_rssi_fspl(rssi), 1])

df = pd.DataFrame(total)
df.to_csv('dataset/2021_04_15/pathloss_v1.csv', header=None, index=None)

train, test = train_test_split(df, test_size=0.3, shuffle=True, random_state=32)
test, valid = train_test_split(test, test_size=0.2, shuffle=True, random_state=32)

train.to_csv('dataset/2021_04_15/pathloss_v1_train.csv', header=None, index=None)
test.to_csv('dataset/2021_04_15/pathloss_v1_test.csv', header=None, index=None)
valid.to_csv('dataset/2021_04_15/pathloss_v1_valid.csv', header=None, index=None)