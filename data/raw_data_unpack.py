import tool.file_io as file_io
import pandas as pd
from model import model_pathloss


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
            x = get_line(directory)
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
    dataset = pd.read_csv(data_path)
    device_list = list(set(list(dataset['mac'])))
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
                        rx_height, tx_antenna_type, tx_board_type] #, x, y]
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
    device01_info = [['nrf_custom', 8, 0, -1.47, 0, 2], ['nrf_dongle', 8, 0, -1, 2, 1], ['cc26x', 5, 0, -1.47, 2, 0]]
    device02_info = [['nrf_custom', 8, 0, -1.47, 0, 2], ['nrf_dongle', 8, 0, -1, 2, 1], ['cc26x', 5, 0, -1.47, 2, 0]]
    device03_info = [['nrf_custom', 8, 0, -1.47, 0, 2], ['nrf_dongle', 8, 0, -1, 2, 1], ['cc26x', 5, 0, -1.47, 2, 0]]
    device04_info = [['proto_v2_FL', 5, 2, 3.6, 1, 3], ['proto_v2_CH', 5, 2, -1.47, 0, 3], ['core_CH', 5, 1, -1.47, 0, 3]]
    device05_info = [['pcb_FL', 5, 0, 3.6, 1, 3]]
    device_info = [device01_info, device02_info, device03_info, device04_info, device05_info]
    # device pack
    device01 = ['f9:9c:e9:5d:b9:6f', 'f9:cd:8d:26:b2:96', 'f8:8a:5e:2d:82:85']  # nrf custom, nrf dongle, cc26x
    device02 = ['d3:0c:28:ba:de:34', 'c6:46:7a:76:ed:6c', 'f8:8a:5e:2d:80:f4']  # nrf custom, nrf dongle, cc26x
    device03 = ['fb:f8:c9:9d:d8:d6', 'ea:33:8f:29:05:0f', 'f8:8a:5e:2d:a4:8a']  # nrf custom, nrf dongle, cc26x
    device04 = ['04:ee:03:74:ae:a5', '04:ee:03:74:b0:30', '04:ee:03:74:b0:3f']  # Flexible, trackingball proto *2, core *1
    device05 = ['04:ee:03:74:ae:dd']  # trackingball pcb
    device = [device01, device02, device03, device04, device05]

    device01_point = [[0, 5], [5, 25], [10, 10], [15, 10], [20, 20], [25, 15]]
    device02_point = [[0, 10], [5, 5], [10, 20], [15, 15], [20, 5], [25, 10]]
    device03_point = [[0, 15], [5, 10], [10, 25], [15, 5], [20, 25], [25, 5]]
    device04_point = [[0, 20], [5, 15], [10, 5], [15, 25], [20, 15], [25, 25]]
    device05_point = [[0, 25], [5, 20], [10, 15], [15, 20], [20, 20], [25, 15]]
    device_point = [device01_point, device02_point, device03_point, device04_point, device05_point]

    device01_point_b = [[0, 25], [5, 15], [10, 10], [15, 20], [20, 25], [25, 25]]
    device02_point_b = [[0, 20], [5, 10], [10, 5], [15, 25], [20, 20], [25, 20]]
    device03_point_b = [[0, 5], [5, 5], [10, 15], [15, 15], [20, 15], [25, 15]]
    device04_point_b = [[0, 10], [5, 25], [10, 20], [15, 10], [20, 15], [25, 10]]
    device05_point_b = [[0, 25], [5, 20], [10, 25], [15, 5], [20, 25], [25, 25]]
    device_point_b = [device01_point_b, device02_point_b, device03_point_b, device04_point_b, device05_point_b]

    input_dir_path = '../dataset/v5/pol03/line'
    # unpack_raw_data(input_dir_path, points=device_point_b, name=device, info=device_info)
    #
    dataset_path = "{}.csv".format(input_dir_path)
    make_training_data(dataset_path, save_path='../dataset/v5/new', main_point=[0, 30])


