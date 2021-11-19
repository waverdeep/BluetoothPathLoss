
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
                                print(points[device_pack_idx][dir_idx])
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


if __name__ == '__main__':
    # antenna type : 0=chip, 1=flexible, 2=pcb
    # board type :
    # device_name, txpower, covered, antenna_gain, antenna_type, board

    ###

    device01_info = [['nrf_custom', 8, 0, -1.47, 0, 2]]
    device02_info = [['nrf_custom', 8, 0, -1.47, 0, 2]]
    device03_info = [['nrf_custom', 8, 0, -1.47, 0, 2]]
    device04_info = [['nrf_custom', 8, 0, -1.47, 0, 2]]
    device05_info = [['cc26x', 5, 0, -1.47, 2, 2]]
    device06_info = [['cc26x', 5, 0, -1.47, 0, 3]]
    device_info = [device01_info, device02_info, device03_info,
                   device04_info, device05_info, device06_info]

    device01 = ['c0:be:b0:c3:d6:3d']
    device02 = ['e4:bf:d3:01:17:37']
    device03 = ['f1:54:2c:92:a8:05']
    device04 = ['f8:8a:5e:45:73:85']
    device05 = ['f8:8a:5e:45:6c:b6']
    device06 = ['04:ee:63:74:b0:30']
    device = [device01, device02, device03, device04, device05, device06]

    # device01_point = [[0, 5], [0, 20], [5, 5], [5, 20], [10, 5], [10, 20], [15, 5], [15, 20], [20, 5], [20, 20],
    #                   [25, 5], [25, 20], [30, 5], [30, 20], [35, 5], [35, 20], [40, 5], [40, 20], [45, 5], [45, 20]]
    # device02_point = [[0, 10], [0, 25], [5, 10], [5, 25], [10, 10], [10, 25], [15, 10], [15, 25], [20, 10], [20, 25],
    #                   [25, 10], [25, 25], [30, 10], [30, 25], [35, 10], [35, 25], [40, 10], [40, 25], [45, 10], [45, 25]]
    # device03_point = [[0, 15], [0, 30], [5, 15], [5, 30], [10, 15], [10, 30], [15, 15], [15, 30], [20, 15], [20, 30],
    #                   [25, 15], [25, 30], [30, 15], [30, 30], [35, 15], [35, 30], [40, 15], [40, 30], [45, 15], [45, 30]]
    # device04_point = [[0, 20], [0, 35], [5, 20], [5, 35], [10, 20], [10, 35], [15, 20], [15, 35], [20, 20], [20, 35],
    #                   [25, 20], [25, 35], [30, 20], [30, 35], [35, 20], [35, 35], [40, 20], [40, 35], [45, 20], [45, 35]]
    # device05_point = [[0, 25], [0, 40], [5, 25], [5, 40], [10, 25], [10, 40], [15, 25], [15, 40], [20, 25], [20, 40],
    #                   [25, 25], [25, 40], [30, 25], [30, 40], [35, 25], [35, 40], [40, 25], [40, 40], [45, 25], [45, 40]]
    # device06_point = [[0, 30], [0, 45], [5, 30], [5, 45], [10, 30], [10, 45], [15, 30], [15, 45], [20, 30], [20, 45],
    #                   [25, 30], [25, 45], [30, 30], [30, 45], [35, 30], [35, 45], [40, 30], [40, 45], [45, 30], [45, 45]]

    device01_point = [[0, 5], [5, 5], [10, 5], [15, 5], [20, 5],
                      [25, 5], [30, 5], [35, 5], [40, 5], [45, 5]]
    device02_point = [[0, 10], [5, 10], [10, 10], [15, 10], [20, 10],
                      [25, 10], [30, 10],  [35, 10], [40, 10],  [45, 10]]
    device03_point = [[0, 15], [5, 15], [10, 15], [15, 15], [20, 15],
                      [25, 15], [30, 15], [35, 15], [40, 15], [45, 15]]
    device04_point = [[0, 20], [5, 20], [10, 20],[15, 20], [20, 20],
                      [25, 20],[30, 20], [35, 20],[40, 20],  [45, 20]]
    device05_point = [[0, 25], [5, 25], [10, 25], [15, 25], [20, 25],
                      [25, 25], [30, 25], [35, 25],[40, 25], [45, 25]]
    device06_point = [[0, 30], [5, 30], [10, 30], [15, 30], [20, 30],
                      [25, 30], [30, 30], [35, 30],  [40, 30], [45, 30]]

    device_point = [device01_point, device02_point, device03_point, device04_point, device05_point, device06_point]
    ###

    input_dir_path = '../dataset/v7/pole04'
    unpack_raw_data(input_dir_path, points=device_point, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    make_training_data(dataset_path, save_path='../dataset/v7/points_pole04', main_point=[50, 30])


