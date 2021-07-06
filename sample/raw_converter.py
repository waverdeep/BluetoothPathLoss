import glob
import os
import natsort
import csv


def get_location(folder):
    x = folder.split('_')[1]
    y = folder.split('_')[2]
    return x, y


f = open("pack_20212-06-14_pol3.csv", 'w', encoding='utf-8')
wr = csv.writer(f)
wr.writerow(['x', 'y', 'mac', 'type', 'RSSi'])

root = 'pack_2021-06-14/pol3'
folder_list = natsort.natsorted(os.listdir(root))

for folder in folder_list:
    print(folder)
    x, y = get_location(folder)
    meter_folder = os.path.join(root, folder)
    print(meter_folder)
    txt_file_path_list = glob.glob(meter_folder + '/*.txt')
    txt_file_path_list = natsort.natsorted(txt_file_path_list)

    for txt_file_path in txt_file_path_list:
        print(txt_file_path)
        file_f = open(txt_file_path, 'r', encoding='utf-8')
        lines = file_f.readlines()

        for line in lines:
            if 'GZ-TrackingBall' in line:
                mac = line.split()[2]
                rssi_type = line.split()[3]
                rssi = line.split()[4]
                wr.writerow([x, y, mac, rssi_type, rssi])
        file_f.close()
f.close()