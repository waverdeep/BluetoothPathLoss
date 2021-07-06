import tool.file_io as file_io


def get_point(dir_name):
    # ex) meter_1_1/
    x = dir_name.split('_')[1]
    y = dir_name.split('_')[2]
    return x, y


def pint_base_raw_data(root_path, dir_list):
    for directory in dir_list:
        x, y = get_point(directory)



if __name__ == '__main__':
    input_dir_path = ''
    directory_list = file_io.get_directory_list(input_dir_path, input_dir_path)


