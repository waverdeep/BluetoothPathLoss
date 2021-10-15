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
def unpack_raw_data(root_path, points=None, name=None, info=None, collection_type='line_base'):
    # "mac", "ADV", "RSSI", "NAME", "CAHNNEL", "TYPE", "TX_POWER", "A_GAIN", "A_TYPE", "BOARD"
    space = []
    # 전체 폴더 00, 05, 10, ....
    dir_list = file_io.get_directory_list(input_dir_path)
    dir_list = file_io.extract_only_directory(dir_list)
    # 전체 폴더 중 하나 접근 00
    for dir_idx, directory in enumerate(dir_list):
        line_base_pack = []
        if collection_type == 'point_base':
            x, y = get_point(directory)
        elif collection_type == 'line_base':
            # x = get_line(directory)
            pass
        # 하나의 라인에 대한 폴더 접근
        dir_path = "{}/{}/".format(root_path, directory)
        # 파일 리스트 추출
        file_list = file_io.get_all_file_path(dir_path, file_extension='txt')
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
                                device_info = info[device_pack_idx][device_idx]
                                split_data.append(channel_info)
                                split_data.append(device_info[0])
                                split_data.append(device_info[1])
                                split_data.append(device_info[2])
                                split_data.append(device_info[3])
                                split_data.append(device_info[4])
                                split_data.append(device_info[5])
                                # print(points[device_pack_idx][dir_idx])
                                device_point_x = points[device_pack_idx][dir_idx][0]
                                device_point_y = points[device_pack_idx][dir_idx][1]
                                split_data.append(device_point_x)
                                split_data.append(device_point_y)
                                space.append(split_data)
                                line_base_pack.append(split_data)

                    # print(split_data)
        pack_pd = pd.DataFrame(line_base_pack, columns=["mac", "ADV", "RSSI", "NAME", "CHANNEL", "TYPE", "TX_POWER",
                                                        "COVERED", "ANTENNA_GAIN", "ANTENNA_TYPE", "BOARD", "X", "Y"])
        save_path = "{}/{}.csv".format(root_path, directory)
        pack_pd.to_csv(save_path, mode='w', index=None)
    pack_all_pd = pd.DataFrame(space, columns=["mac", "ADV", "RSSI", "NAME", "CHANNEL", "TYPE", "TX_POWER",
                                                        "COVERED", "ANTENNA_GAIN", "ANTENNA_TYPE", "BOARD", "X", "Y"])
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
            tx_power = int(data_line['TX_POWER'])
            covered = int(data_line['COVERED'])
            tx_antenna_gain = int(data_line['ANTENNA_GAIN'])
            rx_height = 1.7
            rx_antenna_gain = 16.0
            tx_height = 0.01
            tx_antenna_type = int(data_line['ANTENNA_TYPE'])
            tx_board_type = int(data_line['BOARD'])
            distance = model_pathloss.get_distance(main_point[0], main_point[1], x, y)
            fspl = model_pathloss.fspl_model(rssi)
            new_line = [distance, rssi, fspl, channel, covered, tx_power, tx_antenna_gain, rx_antenna_gain, tx_height,
                        rx_height, tx_antenna_type, tx_board_type, x, y]
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
    # antenna type : 0=chip, 1=flexible, 2=pcb
    # board type :
    # device_name, txpower, covered, antenna_gain, antenna_type, board

    ###
    total = 0

    device01_info = [['nrf_custom', 8, 1, -1.47, 0, 5]] # new nrf
    device02_info = [['nrf_custom', 8, 1, -1.47, 0, 5]]
    device03_info = [['nrf_custom', 8, 1, -1.47, 0, 5]]
    device04_info = [['nrf_custom', 8, 1, -1.47, 0, 5]]
    device05_info = [['cc26x', 5, 1, -1.47, 0, 4]] # new ti
    device06_info = [['cc26x', 5, 2, -1.47, 0, 3]]
    device_info = [device01_info, device02_info, device03_info,
                   device04_info, device05_info, device06_info]

    device01 = ['c0:be:b0:c3:d6:3d'] # nrf/core_only/nospace/
    device02 = ['e4:bf:d3:01:17:37'] # nrf/core_only/space/ -> 뒤짐
    device03 = ['f1:54:2c:92:a8:05'] # nrf/core_only/nospace/
    device04 = ['f8:8a:5e:45:73:85'] # ti/core_only/space/
    device05 = ['f8:8a:5e:45:6c:b6'] # ti/core_only/nospace/
    device06 = ['04:ee:03:74:b0:30'] # ti/trackingball/3layer
    device = [device01, device02, device03, device04, device05, device06]



    # newline
    device01_point = set_label_position(5)
    device02_point = set_label_position(10)
    device03_point = set_label_position(15)
    device04_point = set_label_position(20)
    device05_point = set_label_position(25)
    device06_point = set_label_position(30)
    device_point = [device01_point, device02_point, device03_point, device04_point, device05_point, device06_point]
    ###

    input_dir_path = '../dataset/v7_all/pole01/newline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/points', main_point=[0, 0])

    input_dir_path = '../dataset/v7_all/pole02/newline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/points', main_point=[50, 0])

    input_dir_path = '../dataset/v7_all/pole03/newline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/points', main_point=[0, 30])

    input_dir_path = '../dataset/v7_all/pole04/newline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/points', main_point=[50, 30])

    input_dir_path = '../dataset/v7_all/pole05/newline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/points', main_point=[0, 60])

    input_dir_path = '../dataset/v7_all/pole06/newline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/points', main_point=[50, 60])







    # # reline
    device01_point = set_label_position(20)
    device02_point = set_label_position(25)
    device03_point = set_label_position(30)
    device04_point = set_label_position(35)
    device05_point = set_label_position(40)
    device06_point = set_label_position(45)
    device_point = [device01_point, device02_point, device03_point, device04_point, device05_point, device06_point]

    input_dir_path = '../dataset/v7_all/pole01/reline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/pointsb', main_point=[0, 0])

    input_dir_path = '../dataset/v7_all/pole02/reline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/pointsb', main_point=[50, 0])

    input_dir_path = '../dataset/v7_all/pole03/reline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/pointsb', main_point=[0, 30])

    input_dir_path = '../dataset/v7_all/pole04/reline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/pointsb', main_point=[50, 30])

    input_dir_path = '../dataset/v7_all/pole05/reline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/pointsb', main_point=[0, 60])

    input_dir_path = '../dataset/v7_all/pole06/reline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/pointsb', main_point=[50, 60])





    #
    # leftline
    device01_point = [[0, 5]]
    device02_point = [[0, 10]]
    device03_point = [[0, 15]]
    device04_point = [[0, 20]]
    device05_point = [[0, 25]]
    device06_point = [[0, 30]]
    device_point = [device01_point, device02_point, device03_point, device04_point, device05_point, device06_point]

    input_dir_path = '../dataset/v7_all/pole01/leftline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/left_points', main_point=[0, 0])

    input_dir_path = '../dataset/v7_all/pole02/leftline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/left_points', main_point=[50, 0])

    input_dir_path = '../dataset/v7_all/pole03/leftline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/left_points', main_point=[0, 30])

    input_dir_path = '../dataset/v7_all/pole04/leftline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/left_points', main_point=[50, 30])

    input_dir_path = '../dataset/v7_all/pole05/leftline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/left_points', main_point=[0, 60])

    input_dir_path = '../dataset/v7_all/pole06/leftline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/left_points', main_point=[50, 60])






    #
    # # releftline
    device01_point = [[0, 25]]
    device02_point = [[0, 35]]
    device03_point = [[0, 40]]
    device04_point = [[0, 45]]
    device05_point = [[0, 50]]
    device06_point = [[0, 55]]
    device_point = [device01_point, device02_point, device03_point, device04_point, device05_point, device06_point]

    input_dir_path = '../dataset/v7_all/pole01/releftline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/releft_points', main_point=[0, 0])

    input_dir_path = '../dataset/v7_all/pole02/releftline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/releft_points', main_point=[50, 0])

    input_dir_path = '../dataset/v7_all/pole03/releftline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/releft_points', main_point=[0, 30])

    input_dir_path = '../dataset/v7_all/pole04/releftline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/releft_points', main_point=[50, 30])

    input_dir_path = '../dataset/v7_all/pole05/releftline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/releft_points', main_point=[0, 60])

    input_dir_path = '../dataset/v7_all/pole06/releftline'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    total += make_training_data(dataset_path, save_path='../dataset/v7_all/releft_points', main_point=[50, 60])

    print("total: ", total)





