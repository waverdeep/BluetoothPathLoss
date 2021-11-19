import torch
from model_pack import model_reconst
import numpy as np
from data import data_loader_normal
import time
import math
from trilateration_booster_v1 import trilateration


model_config = {
    "model_type": "DilatedCRNN", "input_size": 3, "sequence_length": 15, "activation": "ReLU",
    "convolution_layer": 4, "bidirectional": False, "hidden_size": 64, "num_layers": 1, "linear_layers": [64, 1],
    "criterion": "MSELoss", "optimizer": "SGD", "dropout_rate": 0.5, "use_cuda": True, "cuda_num": "cuda:0",
    "batch_size": 8, "learning_rate": 0.01, "epoch": 800, "num_workers": 8, "shuffle": True,
    "input_dir": "dataset/v8/type03_train", "tensorboard_writer_path": "runs_2021_11_01",
    "section_message": "Type03-ReLU-sgd0.01-T71", "checkpoint_dir": "checkpoints/2021_11_01",
    "checkpoint_path": "test_checkpoints/Type01-ReLU-sgd0.0025-T33_epoch_280.pt"
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
    inference_model = model_reconst.model_load(model_config)
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
        "dataset/v8/test1_type01_point_f8-8a-5e-45-75-9b_38_40_30/f8-8a-5e-45-75-9b_38_40_30_pol50-0.csv", # pole2
        "", # pole3
        "dataset/v8/test1_type01_point_f8-8a-5e-45-75-9b_38_40_30/f8-8a-5e-45-75-9b_38_40_30_pol50-30.csv", # pole4
    ]

    inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[1])
    inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[3])



    start_time = time.time()
    print(time.time() - start_time)