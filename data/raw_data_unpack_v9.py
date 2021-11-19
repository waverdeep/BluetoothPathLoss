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
    # print(dir_list)
    # 전체 폴더 중 하나 접근 00
    for dir_idx, directory in enumerate(dir_list):
        line_base_pack = []
        # 하나의 라인에 대한 폴더 접근
        dir_path = "{}/{}/".format(root_path, directory)
        # 파일 리스트 추출
        file_list = file_io.get_all_file_path(dir_path, file_extension='txt')
        # print(file_list)
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

                    # print(split_data)

        pack_pd = pd.DataFrame(line_base_pack, columns=["mac", "ADV", "RSSI", "NAME", "CHANNEL", "X", "Y"])
        save_path = "{}/{}.csv".format(root_path, directory)
        pack_pd.to_csv(save_path, mode='w', index=None)
    pack_all_pd = pd.DataFrame(space, columns=["mac", "ADV", "RSSI", "NAME", "CHANNEL", "X", "Y"])
    save_path = "{}.csv".format(root_path)
    pack_all_pd.to_csv(save_path, mode='w', index=None)


def make_training_data(data_path, save_path, main_point):
    new_all_pack = []
    # print(data_path)
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
            new_line = [distance, rssi, fspl, channel, x, y]
            lines.append(new_line)
            new_all_pack.append(new_line)
        if len(lines) > 15:
            df = pd.DataFrame(lines)
            name = mac.replace(":", '-')
            path = "{}/{}_{}-{}_{}.csv".format(save_path, name, main_point[0], main_point[1], channel)
            df.to_csv(path, header=None, index=None)
    # print(len(new_all_pack))
    return len(new_all_pack)


def set_label_position(tip):
    device_point = []
    for i in np.arange(0, 65, 5).tolist():
        device_point.append([tip, i])
    return device_point


if __name__ == '__main__':
    device = [  # 간격이 떨어져 있지 않은 것
        ['f8:8a:5e:45:73:b0'],  # 3
        ['f8:8a:5e:45:77:07'],  # 4
        ['f8:8a:5e:45:74:8c'],  # 5
        ['f8:8a:5e:45:73:e5'],  # 6
        ['f8:8a:5e:45:72:ea'],  # 7
        ['f8:8a:5e:45:74:bc'],  # 8
        ['f8:8a:5e:45:74:9e'],  # 9
        ['f8:8a:5e:45:71:80'],  # 10
        ['f8:8a:5e:45:73:fd'],  # 11
        ['f8:8a:5e:45:71:85'],  # 12
        ['f8:8a:5e:45:74:c8'],  # 13
        ['f8:8a:5e:45:6c:c1'],  # 17
        ['f8:8a:5e:45:75:ad'],  # 18
    ]

    device03 = [[i, 0] for i in range(5, 50, 5)]
    device04 = [[i, 5] for i in range(5, 50, 5)]
    device05 = [[i, 10] for i in range(5, 50, 5)]
    device06 = [[i, 15] for i in range(5, 50, 5)]
    device07 = [[i, 20] for i in range(5, 50, 5)]
    device08 = [[i, 25] for i in range(5, 50, 5)]
    device09 = [[i, 30] for i in range(5, 50, 5)]
    device10 = [[i, 5] for i in range(10, 50, 5)]
    device11 = [[i, 10] for i in range(10, 50, 5)]
    device12 = [[i, 15] for i in range(10, 50, 5)]
    device13 = [[i, 20] for i in range(10, 50, 5)]
    device17 = [[i, 25] for i in range(10, 50, 5)]
    device18 = [[i, 30] for i in range(10, 50, 5)]
    device_point = [device03, device04, device05, device06, device07, device08, device09, device10,
                    device11, device12, device13, device17, device18]

    total = 0
    input_dir_path = '../dataset/v9/pole01/type01'
    unpack_raw_data(input_dir_path, points=device_point, name=device)

    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v9/point_type01', main_point=[0, 0])

    input_dir_path = '../dataset/v9/pole02/type01'
    unpack_raw_data(input_dir_path, points=device_point, name=device)

    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v9/point_type01', main_point=[50, 0])

    input_dir_path = '../dataset/v9/pole03/type01'
    unpack_raw_data(input_dir_path, points=device_point, name=device)

    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v9/point_type01', main_point=[0, 30])

    input_dir_path = '../dataset/v9/pole04/type01'
    unpack_raw_data(input_dir_path, points=device_point, name=device)

    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v9/point_type01', main_point=[50, 30])
    print("total: ", total)


    device03 = list(reversed([[i, 15] for i in range(35, 41, 1)]))+list(reversed([[i, 10] for i in range(15, 21, 1)]))
    device04 = list(reversed([[i, 16] for i in range(35, 41, 1)]))+list(reversed([[i, 11] for i in range(15, 21, 1)]))
    device05 = list(reversed([[i, 17] for i in range(35, 41, 1)]))+list(reversed([[i, 12] for i in range(15, 21, 1)]))
    device06 = list(reversed([[i, 18] for i in range(35, 41, 1)]))+list(reversed([[i, 13] for i in range(15, 21, 1)]))
    device07 = list(reversed([[i, 19] for i in range(35, 41, 1)]))+list(reversed([[i, 14] for i in range(15, 21, 1)]))
    device08 = list(reversed([[i, 20] for i in range(35, 41, 1)]))+list(reversed([[i, 15] for i in range(15, 21, 1)]))
    device09 = list(reversed([[i, 21] for i in range(35, 41, 1)]))+list(reversed([[i, 16] for i in range(15, 21, 1)]))
    device10 = list(reversed([[i, 22] for i in range(35, 41, 1)]))+list(reversed([[i, 17] for i in range(15, 21, 1)]))
    device11 = list(reversed([[i, 23] for i in range(35, 41, 1)]))+list(reversed([[i, 18] for i in range(15, 21, 1)]))
    device12 = list(reversed([[i, 24] for i in range(35, 41, 1)]))+list(reversed([[i, 19] for i in range(15, 21, 1)]))
    device13 = list(reversed([[i, 25] for i in range(35, 41, 1)]))+list(reversed([[i, 20] for i in range(15, 21, 1)]))
    device17 = list(reversed([[i, 0] for i in range(4, 16, 1)]))
    device18 = list(reversed([[i, 0] for i in range(3, 15, 1)]))
    device_point = [device03, device04, device05, device06, device07, device08, device09, device10,
                    device11, device12, device13, device17, device18]

    total = 0
    input_dir_path = '../dataset/v9/pole01/type02'
    unpack_raw_data(input_dir_path, points=device_point, name=device)

    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v9/point_type02', main_point=[0, 0])

    input_dir_path = '../dataset/v9/pole02/type02'
    unpack_raw_data(input_dir_path, points=device_point, name=device)

    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v9/point_type02', main_point=[50, 0])

    input_dir_path = '../dataset/v9/pole03/type02'
    unpack_raw_data(input_dir_path, points=device_point, name=device)

    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v9/point_type02', main_point=[0, 30])

    input_dir_path = '../dataset/v9/pole04/type02'
    unpack_raw_data(input_dir_path, points=device_point, name=device)

    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v9/point_type02', main_point=[50, 30])
    print("total: ", total)