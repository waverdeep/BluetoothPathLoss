import json
from tool import file_io


if __name__ == '__main__':
    file_path = '../configurations/v5_v3/config_crnn.json'
    index_num = 31
    with open(file_path) as f:
        json_data = json.load(f)
        print(json_data[index_num])


