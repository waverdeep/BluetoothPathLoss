import numpy as np
import tool.file_io as file_io
import pandas as pd
from model_pack import model_pathloss


def get_point(dir_name):
    # ex) meter_1_1/
    x = dir_name.split('_')[1]
    y = dir_name.split('_')[2]
    return x, y


def get_line(dir_name):
    # ex) info_line_00
    x = int(dir_name.split('_')[2])
    return x


def split_raw_data_line(line, check_label='ADV_IND'):
    if check_label in line:
        truncated = line.split(' ')
        truncated = ''.join(truncated)
        truncated = truncated.split('||')
        truncated[3] = truncated[3].replace("\n", "")
        return truncated
    return ''


# type : point_base, line_base
def unpack_raw_data(root_path, points=None, name=None, collection_type='line_base'):
    # "mac", "ADV", "RSSI", "NAME", "CAHNNEL", "TYPE", "TX_POWER", "A_GAIN", "A_TYPE", "BOARD"
    space = []
    # 전체 폴더 00, 05, 10, ....
    dir_list = file_io.get_directory_list(input_dir_path)
    dir_list = file_io.extract_only_directory(dir_list)
    print(dir_list)
    # 전체 폴더 중 하나 접근 00
    for dir_idx, directory in enumerate(dir_list):
        line_base_pack = []
        # 하나의 라인에 대한 폴더 접근
        dir_path = "{}/{}/".format(root_path, directory)
        # 파일 리스트 추출
        file_list = file_io.get_all_file_path(dir_path, file_extension='txt')
        print(file_list)
        # 하나의 파일 열어서 가공
        for file_idx, file in enumerate(file_list):
            channel_info = file_io.get_pure_filename(file).split('_')[1]
            lines = file_io.read_txt_file(file)
            for line in lines:
                # 0: mac
                # 1: ADV
                # 2: RSSI
                # 3: Name
                split_data = split_raw_data_line(line)
                if split_data != '':
                    for device_pack_idx, device_pack in enumerate(name):
                        for device_idx, device_name in enumerate(device_pack):
                            if device_name in split_data[0]:
                                if len(points[device_pack_idx][dir_idx]) == 0:
                                    continue
                                split_data.append(channel_info)
                                # print("device pack idx : ", points[device_pack_idx])
                                # print("point : ", points[device_pack_idx][dir_idx])
                                # print("data : ", split_data)
                                device_point_x = points[device_pack_idx][dir_idx][0]
                                device_point_y = points[device_pack_idx][dir_idx][1]
                                split_data.append(device_point_x)
                                split_data.append(device_point_y)
                                space.append(split_data)
                                line_base_pack.append(split_data)

                    print(split_data)

        pack_pd = pd.DataFrame(line_base_pack, columns=["mac", "ADV", "RSSI", "NAME", "CHANNEL", "X", "Y"])
        save_path = "{}/{}.csv".format(root_path, directory)
        pack_pd.to_csv(save_path, mode='w', index=None)
    pack_all_pd = pd.DataFrame(space, columns=["mac", "ADV", "RSSI", "NAME", "CHANNEL", "X", "Y"])
    save_path = "{}.csv".format(root_path)
    pack_all_pd.to_csv(save_path, mode='w', index=None)


def make_training_data(data_path, save_path, main_point):
    new_all_pack = []
    print(data_path)
    dataset = pd.read_csv(data_path)
    device_list = list(set(list(dataset['mac'])))
    print(device_list)
    dataset_pack = []
    for device_mac in device_list:
        temp = pd.DataFrame(dataset[dataset['mac'] == device_mac].to_dict())
        c37 = pd.DataFrame(temp[temp['CHANNEL'] == 37].to_dict())
        c38 = pd.DataFrame(temp[temp['CHANNEL'] == 38].to_dict())
        c39 = pd.DataFrame(temp[temp['CHANNEL'] == 39].to_dict())
        dataset_pack.append(c37)
        dataset_pack.append(c38)
        dataset_pack.append(c39)
    for data_idx, data in enumerate(dataset_pack):
        lines = []
        for line_idx, data_line in data.iterrows():
            x = int(data_line['X'])
            y = int(data_line['Y'])
            mac = data_line['mac']
            rssi = int(data_line['RSSI'])
            channel = int(data_line['CHANNEL'])
            distance = model_pathloss.get_distance(main_point[0], main_point[1], x, y)
            fspl = model_pathloss.fspl_model(rssi)
            new_line = [distance, rssi, fspl, channel]#, x, y]
            lines.append(new_line)
            new_all_pack.append(new_line)
        if len(lines) > 15:
            df = pd.DataFrame(lines)
            name = mac.replace(":", '-')
            path = "{}/{}_{}-{}_{}.csv".format(save_path, name, main_point[0], main_point[1], channel)
            df.to_csv(path, header=None, index=None)
    print(len(new_all_pack))
    return len(new_all_pack)


def set_label_position(tip):
    device_point = []
    for i in np.arange(0, 65, 5).tolist():
        device_point.append([tip, i])
    return device_point


if __name__ == '__main__':
    device = [  # 생보드
        ['04:ee:03:74:ae:ad'],  # 1
        ['04:ee:03:74:ae:ef'],  # 2
        ['04:ee:03:74:b0:10'],  # 3
        ['04:ee:03:74:b0:20'],  # 4
        ['04:ee:03:74:ae:e0'],  # 5
        ['04:ee:03:74:ae:bd'],  # 6
    ]

    device01 = [[0, 5], [5, 10], [10, 0], [15, 0], [20, 0], [25, 5], [30, 0], [35, 0], [40, 0], [45, 0], [50, 5]]
    device02 = [[0, 10], [5, 15], [10, 5], [15, 5], [20, 5], [25, 10], [30, 5], [35, 5], [40, 5], [45, 5], [50, 10]]
    device03 = [[0, 15], [5, 20], [10, 10], [15, 10], [20, 10], [25, 15], [30, 10], [35, 10], [40, 10], [45, 10], [50, 15]]
    device04 = [[0, 20], [5, 25], [10, 15], [15, 15], [20, 15], [25, 20], [30, 15], [35, 15], [40, 15], [45, 15], [50, 20]]
    device05 = [[0, 25], [5, 30], [10, 20], [15, 20], [20, 20], [25, 25], [30, 20], [35, 20], [40, 20], [45, 20], [50, 25]]
    device06 = [[5, 0], [10, 0], [10, 25], [15, 25], [20, 25], [25, 30], [30, 25], [35, 25], [40, 25], [45, 25], [45, 25]]

    device_point = [device01, device02, device03, device04, device05, device06]

    total = 0
    input_dir_path = '../dataset/v8/pole01/type03'
    unpack_raw_data(input_dir_path, points=device_point, name=device)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v8/type03_train_l', main_point=[0, 0])

    total = 0
    input_dir_path = '../dataset/v8/pole02/type03'
    unpack_raw_data(input_dir_path, points=device_point, name=device)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v8/type03_train_l', main_point=[50, 0])

    total = 0
    input_dir_path = '../dataset/v8/pole03/type03'
    unpack_raw_data(input_dir_path, points=device_point, name=device)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v8/type03_train_l', main_point=[0, 30])

    total = 0
    input_dir_path = '../dataset/v8/pole04/type03'
    unpack_raw_data(input_dir_path, points=device_point, name=device)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v8/type03_train_l', main_point=[50, 30])

    # device = [  # 간격이 떨어져 있지 않은 것
    #     ['f8:8a:5e:45:72:bc'],  # 1
    #     ['f8:8a:5e:45:71:ab'],  # 2
    #     ['f8:8a:5e:45:6c:cd'],  # 3
    #     ['f8:8a:5e:45:6c:e1'],  # 4
    #     ['f8:8a:5e:45:77:3b'],  # 5
    #     ['f8:8a:5e:45:75:84'],  # 6
    #     ['f8:8a:5e:45:75:8f'],  # 7
    #     ['f8:8a:5e:45:75:e6'],  # 8
    #     ['f8:8a:5e:45:72:a9'],  # 9
    #     ['f8:8a:5e:45:73:b3'],  # 10
    #     ['f8:8a:5e:45:77:2a'],  # 11
    #     ['f8:8a:5e:45:77:08'],  # 12
    #     ['f8:8a:5e:45:73:8e'],  # 13
    #     ['f8:8a:5e:45:77:32'],  # 14
    # ]
    #
    # device01 = [[0, 5], [10, 10], [20, 10], [30, 10], [40, 10]]
    # device02 = [[0, 10], [10, 15], [20, 15], [], [40, 15]]
    # device03 = [[0, 15], [10, 20], [20, 20], [30, 20], [40, 20]]
    # device04 = [[0, 20], [10, 25], [20, 25], [30, 25], [40, 25]]
    # device05 = [[0, 25], [10, 30], [20, 30], [30, 30], [40, 30]]
    # device06 = [[5, 0], [15, 0], [25, 0], [35, 0], []]
    # device07 = [[5, 5], [15, 5], [25, 5], [35, 5], []]
    # device08 = [[5, 10], [15, 10], [25, 10], [], []]
    # device09 = [[5, 15], [15, 15], [25, 15], [35, 15], [45, 15]]
    # device10 = [[5, 20], [15, 20], [25, 20], [35, 20], [45, 20]]
    # device11 = [[5, 25], [15, 25], [25, 25], [35, 25], [45, 25]]
    # device12 = [[5, 30], [20, 5], [25, 30], [35, 30], [45, 30]]
    # device13 = [[10, 0], [20, 0], [30, 0], [40, 0], [50, 10]]
    # device14 = [[10, 5], [15, 30], [30, 5], [], [50, 15]]
    #
    # device_point = [device01, device02, device03, device04, device05, device06, device07, device08, device09, device10,
    #                 device11, device12, device13, device14]
    #
    # total = 0
    # input_dir_path = '../dataset/v8/pole01/type02'
    # unpack_raw_data(input_dir_path, points=device_point, name=device)
    # #
    # dataset_path = "{}.csv".format(input_dir_path)
    # total += make_training_data(dataset_path, save_path='../dataset/v8/type02_train', main_point=[0, 0])
    #
    # input_dir_path = '../dataset/v8/pole02/type02'
    # unpack_raw_data(input_dir_path, points=device_point, name=device)
    # #
    # dataset_path = "{}.csv".format(input_dir_path)
    # total += make_training_data(dataset_path, save_path='../dataset/v8/type02_train', main_point=[50, 0])
    #
    # input_dir_path = '../dataset/v8/pole03/type02'
    # unpack_raw_data(input_dir_path, points=device_point, name=device)
    # #
    # dataset_path = "{}.csv".format(input_dir_path)
    # total += make_training_data(dataset_path, save_path='../dataset/v8/type02_train', main_point=[0, 30])
    #
    # input_dir_path = '../dataset/v8/pole04/type02'
    # unpack_raw_data(input_dir_path, points=device_point, name=device)
    # #
    # dataset_path = "{}.csv".format(input_dir_path)
    # total += make_training_data(dataset_path, save_path='../dataset/v8/type02_train', main_point=[50, 30])
    #
    # new line new pack
    # device = [  # 간격이 떨어져 있는 것
    #     ['f8:8a:5e:45:75:9b'],  # 1
    #     ['f8:8a:5e:45:75:c5'],  # 2
    #     ['f8:8a:5e:45:73:b0'],  # 3
    #     ['f8:8a:5e:45:77:07'],  # 4
    #     ['f8:8a:5e:45:74:8c'],  # 5
    #     ['f8:8a:5e:45:73:e5'],  # 6
    #     ['f8:8a:5e:45:72:ea'],  # 7
    #     ['f8:8a:5e:45:74:bc'],  # 8
    #     ['f8:8a:5e:45:74:9e'],  # 9
    #     ['f8:8a:5e:45:71:80'],  # 10
    #     ['f8:8a:5e:45:73:fd'],  # 11
    #     ['f8:8a:5e:45:71:85'],  # 12
    #     ['f8:8a:5e:45:74:c8'],  # 13
    #     ['f8:8a:5e:45:75:cf'],  # 14
    #     ['f8:8a:5e:45:77:06'],  # 15
    #     ['f8:8a:5e:45:77:35'],  # 16
    #     ['f8:8a:5e:45:6c:c1'],  # 17
    #     ['f8:8a:5e:45:75:ad'],  # 18
    #     ['f8:8a:5e:45:77:0e'],  # 19
    #     ['f8:8a:5e:45:72:f8'],  # 20
    # ]
    #
    # device01 = [[0, 5], [15, 5], [30, 0], [40, 30]]
    # device02 = [[0, 10], [15, 10], [30, 5], []]
    # device03 = [[0, 15], [15, 15], [30, 10], []]
    # device04 = [[0, 20], [15, 20], [], []]
    # device05 = [[0, 25], [15, 25], [30, 20], [45, 15]]
    # device06 = [[5, 0], [15, 30], [30, 25], [45, 20]]
    # device07 = [[5, 5], [20, 0], [30, 30], [45, 25]]
    # device08 = [[5, 10], [20, 5], [35, 0], [45, 30]]
    # device09 = [[5, 15], [20, 10], [35, 5], []]
    # device10 = [[5, 20], [20, 15], [], [50, 15]]
    # device11 = [[5, 25], [20, 20], [35, 15], [50, 20]]
    # device12 = [[5, 30], [20, 25], [35, 20], []]
    # device13 = [[10, 0], [20, 30], [35, 25], [50, 25]]
    # device14 = [[10, 5], [25, 0], [35, 30], [0, 5]]
    # device15 = [[10, 10], [25, 5], [40, 0], [0, 10]]
    # device16 = [[10, 15], [25, 10], [], [0, 15]]
    # device17 = [[10, 20], [25, 15], [40, 10], [0, 20]]
    # device18 = [[10, 25], [25, 20], [40, 15], [0, 25]]
    # device19 = [[10, 30], [25, 25], [40, 20], [5, 0]]
    # device20 = [[15, 0], [25, 30], [40, 25], [5, 5]]
    #
    # device_point = [device01, device02, device03, device04, device05, device06, device07, device08, device09, device10,
    #                 device11, device12, device13, device14, device15, device16, device17, device18, device19, device20]
    # ###
    #
    # total = 0
    # input_dir_path = '../dataset/v8/pole01/type01'
    # unpack_raw_data(input_dir_path, points=device_point, name=device)
    # #
    # dataset_path = "{}.csv".format(input_dir_path)
    # total += make_training_data(dataset_path, save_path='../dataset/v8/type01_train', main_point=[0, 0])
    #
    # input_dir_path = '../dataset/v8/pole02/type01'
    # unpack_raw_data(input_dir_path, points=device_point, name=device)
    # #
    # dataset_path = "{}.csv".format(input_dir_path)
    # total += make_training_data(dataset_path, save_path='../dataset/v8/type01_train', main_point=[50, 0])
    #
    # input_dir_path = '../dataset/v8/pole03/type01'
    # unpack_raw_data(input_dir_path, points=device_point, name=device)
    # #
    # dataset_path = "{}.csv".format(input_dir_path)
    # total += make_training_data(dataset_path, save_path='../dataset/v8/type01_train', main_point=[0, 30])
    #
    # input_dir_path = '../dataset/v8/pole04/type01'
    # unpack_raw_data(input_dir_path, points=device_point, name=device)
    # #
    # dataset_path = "{}.csv".format(input_dir_path)
    # total += make_training_data(dataset_path, save_path='../dataset/v8/type01_train', main_point=[50, 30])
    # #
    # #
    #
    #
    #
