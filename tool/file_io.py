import glob
import natsort
import os


def get_all_file_path(input_dir, file_extension):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp


def get_filename(input_filepath):
    return input_filepath.split('/')[-1]


def get_pure_filename(input_filepath):
    temp = input_filepath.split('/')[-1]
    return temp.split('.')[0]


def create_directory(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except OSError:
        print('Error : Creating directory: '+dir_path)


def get_directory_list(dir_path, sort=True):
    if sort:
        return natsort.natsorted(os.listdir(dir_path))
    else:
        return os.listdir(dir_path)