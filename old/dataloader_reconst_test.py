import numpy as np
from tool import file_io
import pandas as pd


input_dir = '../dataset/v9/loader_test'
input_size = 20

# 파일들이 저장되었는 경로를 받아 파일 리스트를 얻어냄
file_list = file_io.get_all_file_path(input_dir, file_extension='csv')
# csv에 있는 모든 데이터를 다 꺼내서 넘파이로 만듬
addition_dataset = []
setup_dataset = None
for idx, file in enumerate(file_list):
    addition_dataset.append(pd.read_csv(file).to_numpy())

div_meter_pack = []
rnn_dataset = []
for n_idx, pack in enumerate(addition_dataset):
    label = pack[:, 0].tolist()
    label = list(set(label))
    temp_pack = pd.DataFrame(pack)
    for key in label:
        div_meter_pack.append(temp_pack[temp_pack[0] == key].to_numpy())

for n_idx, pack in enumerate(div_meter_pack):
    print(len(pack))
    if len(pack) < 30:
        temp = pack.tolist()
        temp = temp * (int(30/len(pack))+6)
        pack = np.array(temp)
    print("modi", len(pack))
    for i in range(len(pack) - input_size):
        rnn_dataset.append(pack[i:i + input_size])

    # if various_input is True:
    #     for i in range(len(pack)-input_size):
    #         rnn_dataset.append(pack[i:i+np.random.randint(input_size-7)+7])