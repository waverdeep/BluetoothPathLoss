import torch
from model_pack import model
import numpy as np
from data import data_loader_normal
import time
import math
from trilateration_booster_v1 import trilateration


model_config = {
    "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "LeakyReLU",
    "convolution_layer": 2, "bidirectional": False, "hidden_size": 256, "num_layers": 1,
    "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
    "use_cuda": True, "batch_size": 8000, "learning_rate": 0.0005, "epoch": 7000,
    "num_workers": 8, "shuffle": True, "input_dir": "dataset/v7_all/point_all_v6", "cuda_num": 'cuda:0',
    "checkpoint_path": "test_checkpoints/quick_01-09_Custom_CRNN_AdamW_LeakyReLU_0.0005_sl15_999_epoch_6999.pt"
}


def inference(nn_model, model_config, file):
    nn_model = nn_model
    inference_dataloader = \
        data_loader.load_path_loss_with_detail_inference_dataset(file, 'CRNN',
                                                                 batch_size=model_config['batch_size'])
    total_pred = []
    with torch.no_grad():
        for i, data in enumerate(inference_dataloader):
            print(data.shape)
            x_data = data.transpose(1, 2)
            x_data = x_data.cuda()
            y_pred = nn_model(x_data).reshape(-1).cpu().numpy()
            total_pred.extend(y_pred)
        total_pred = np.array(total_pred)
        print('mean : ', total_pred.mean())
        print('median : ', np.median(total_pred))
        print('max : ', total_pred.max())
        print('min : ', total_pred.min())
        return {'mean': total_pred.mean(), 'median': np.median(total_pred),
                'max': total_pred.max(), 'min': total_pred.min()}


if __name__ == '__main__':
    inference_model = model.model_load(model_config)
    pole_line_count = 2
    height = 30
    width = 50
    pole_point = []
    circles = []
    for i in range(pole_line_count):
        pole_point.append([i*height, 0])
        pole_point.append([i*height, width])

    pole_data_path = [
        "", # pole1
        "", # pole2
        "", # pole3
        "", # pole4
        "", # pole5
        "", # pole6
    ]

    if pole_line_count*2 != len(pole_data_path):
        print("unmatched pole_line_count and pole_data_path")
        exit(0)
    loop_count = pole_line_count * 2
    for i in range(loop_count):
        if pole_data_path == "":
            break
        # inference(nn_model=inference_model, model_config=model_config, file)



    start_time = time.time()
    print(time.time() - start_time)