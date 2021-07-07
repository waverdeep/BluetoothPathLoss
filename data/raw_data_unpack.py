import tool.file_io as file_io


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
        return truncated
    return ''



# type : point_base, line_base
def unpack_raw_data(root_path, points=None, name=None, info=None, collection_type='line_base'):
    dir_list = file_io.get_directory_list(input_dir_path)
    for directory in dir_list:
        if collection_type == 'point_base':
            x, y = get_point(directory)
        elif collection_type == 'line_base':
            x = get_line(directory)

        dir = "{}{}/".format(root_path, directory)
        file_list = file_io.get_all_file_path(dir, file_extension='txt')
        for file_idx, file in enumerate(file_list):
            lines = file_io.read_txt_file(file)





if __name__ == '__main__':
    device01_info = ['nrf_custom', 'nrf_dongle', 'cc26x']
    device02_info = ['nrf_custom', 'nrf_dongle', 'cc26x']
    device03_info = ['nrf_custom', 'nrf_dongle', 'cc26x']
    device04_info = ['proto_v2_FL', 'proto_v2_CH', 'core_CH']
    device05_info = ['pcb_FL']
    device_info = [device01_info, device02_info, device03_info, device04_info, device05_info]
    # device pack
    device01 = ['f9:9c:e9:5d:b9:6f', 'f9:cd:8d:26:b2:96', 'f8:8a:5e:2d:82:85']  # nrf custom, nrf dongle, cc26x
    device02 = ['d3:0c:28:ba:de:34', 'c6:46:7a:76:ed:6c', 'd3:0c:28:ba:de:34']  # nrf custom, nrf dongle, cc26x
    device03 = ['fb:f8:c9:9d:d8:d6', 'ea:33:8f:29:05:0f', 'f8:8a:5e:2d:a4:8a']  # nrf custom, nrf dongle, cc26x
    device04 = ['04:ee:03:74:ae:a5', '04:ee:03:74:b0:30',
                '04:ee:03:74:b0:3f']  # Flexible, trackingball proto *2, core *1
    device05 = ['04:ee:03:74:ae:dd']  # trackingball pcb
    device = [device01, device02, device03, device04, device05]

    device01_point = [[0, 5], [5, 25], [10, 10], [15, 10], [20, 20], [25, 15]]
    device02_point = [[0, 10], [5, 5], [10, 20], [15, 15], [20, 5], [25, 10]]
    device03_point = [[0, 15], [5, 10], [10, 25], [15, 5], [20, 25], [25, 5]]
    device04_point = [[0, 20], [5, 14], [10, 5], [15, 24], [20, 15], [25, 25]]
    device05_point = [[0, 25], [5, 20], [10, 15], [15, 20], [20, 20], [25, 15]]
    device_point = [device01_point, device02_point, device03_point, device04_point, device05_point]

    device01_point_b = [[0, 25], [5, 15], [10, 10], [15, 20], [20, 25], [25, 25]]
    device02_point_b = [[0, 20], [5, 10], [10, 5], [15, 25], [20, 20], [25, 20]]
    device03_point_b = [[0, 5], [5, 5], [10, 15], [15, 15], [20, 15], [25, 15]]
    device04_point_b = [[0, 10], [5, 25], [10, 20], [15, 10], [20, 15], [25, 10]]
    device05_point_b = [[0, 25], [5, 20], [10, 25], [15, 5], [20, 25], [25, 25]]
    device_point_b = [device01_point_b, device02_point_b, device03_point_b, device04_point_b, device05_point_b]

    input_dir_path = '../dataset/v5/pol01/line/'
    unpack_raw_data(input_dir_path)


