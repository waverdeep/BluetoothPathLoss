# gray (type01)
type01 = ['ee:01:19:a0:ba:9b', 'cc:cf:1a:8f:35:8b', 'd3:0c:28:ba:de:34']
# white (type02)
type02 = ['fc:e6:04:1b:5c:0d', 'f9:cd:8d:26:b2:96', 'fb:f8:c9:9d:d8:d6']
# black (type03)
type03 = ['e2:10:4f:cd:ef:db', 'f9:55:1c:cb:f0:4b', 'f9:9c:e9:5d:b9:6f']
# custom (type04)
type04 = ['04:ee:03:74:ae:e7', '04:ee:03:74:ae:f5', '04:ee:03:74:b0:3f']

type01_point = [[5, 30], [10, 25], [15, 20], [20, 15], [25, 10], [30, 5]]
type02_point = [[5, 10], [10, 10], [15, 15], [20, 20], [25, 25], [30, 15]]
type03_point = [[5, 5], [10, 5], [15, 5], [20, 5], [25, 15], [30, 10]]
type04_point = [[5, 25], [10, 20], [15, 10], [20, 10], [25, 0], [30, 20]]

import glob
import os
import natsort
import csv

f = open("rssi_pack_pol3_2021_06_22.csv", 'w', encoding='utf-8')
wr = csv.writer(f)
wr.writerow(['x', 'y', 'mac', 'type', 'RSSi'])

# 입력 파일 경로
# pol 경로를 줘야x함
root = 'pack/rssi_pack_pol3_2021_06_22'
folder_list = natsort.natsorted(os.listdir(root))

# pol에 있는 모든 폴더에 대해서 경로를 만듬
# folder_list = [meter_line_05, meter_line_10 ...]
for i, folder in enumerate(folder_list):
    print(folder)
    meter_folder = os.path.join(root, folder)
    print(meter_folder)
    print(type01_point[i])
    txt_file_path_list = glob.glob(meter_folder + '/*.txt')
    txt_file_path_list = natsort.natsorted(txt_file_path_list)

    # 모든 text에 대해서 다 확인
    # txt_file_path_list = meter_line에 있는 모든 txt파일을 담고있음
    for txt_file_path in txt_file_path_list:
        print(txt_file_path)
        file_f = open(txt_file_path, 'r', encoding='utf-8')
        lines = file_f.readlines()
        file_f.close()

        # 각 type별로 구분해서 csv파일로 써주기
        for line in lines:
            for type1 in type01:
                if type1 in line:
                    x = type01_point[i][0]
                    y = type01_point[i][1]
                    mac = line.split()[2]
                    rssi_type = line.split()[3]
                    rssi = line.split()[4]
                    wr.writerow([x, y, mac, rssi_type, rssi])
            for type2 in type02:
                if type2 in line:
                    x = type02_point[i][0]
                    y = type02_point[i][1]
                    mac = line.split()[2]
                    rssi_type = line.split()[3]
                    rssi = line.split()[4]
                    wr.writerow([x, y, mac, rssi_type, rssi])
            for type3 in type03:
                if type3 in line:
                    x = type03_point[i][0]
                    y = type03_point[i][1]
                    mac = line.split()[2]
                    rssi_type = line.split()[3]
                    rssi = line.split()[4]
                    wr.writerow([x, y, mac, rssi_type, rssi])
            for type4 in type04:
                if type4 in line:
                    x = type04_point[i][0]
                    y = type04_point[i][1]
                    mac = line.split()[2]
                    rssi_type = line.split()[3]
                    rssi = line.split()[4]
                    wr.writerow([x, y, mac, rssi_type, rssi])

f.close()