import torch
from model_pack import model_reconst
import numpy as np
from data import data_loader_normal
import time
import math
from trilateration_booster_v1 import trilateration


model_config = {
    "model_type": "DilatedCRNNSmallV8", "input_size": 3, "sequence_length": 20, "activation": "ReLU",
    "convolution_layer": 4, "bidirectional": False, "hidden_size": 128, "num_layers": 1,
    "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "SGD", "dropout_rate": 0.5,
    "use_cuda": True, "cuda_num": "cuda:0", "batch_size": 8, "learning_rate": 0.005,
    "epoch": 500, "num_workers": 8, "shuffle": True, "input_dir": "dataset/v9/points_v8-v9-weak",
    "tensorboard_writer_path": "runs_2021_11_11",
    "section_message": "Trial01_v8-v9-weak-smallv8-sgd0.005-seq10",
    "checkpoint_dir": "checkpoints/2021_11_11",
    "checkpoint_path": "checkpoints/2021_11_11/modi6_reopen_Trial01_v8-v9-smallv8-adamw0.001-seq20_epoch_78.pt"
}


def inference(nn_model, model_config, file):
    nn_model = nn_model
    inference_dataloader = \
        data_loader.load_path_loss_with_detail_inference_dataset(file, 'CRNN',
                                                                 batch_size=model_config['batch_size'],
                                                                 input_size=model_config['sequence_length'])
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
        "dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-75-c5_39_15_10/f8-8a-5e-45-75-c5_39_15_10_pol0-0.csv",
        # pole1
        "dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-75-c5_39_15_10/f8-8a-5e-45-75-c5_39_15_10_pol50-0.csv",
        # pole2
        "dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-75-c5_39_15_10/f8-8a-5e-45-75-c5_39_15_10_pol0-30.csv",
        # pole3
        "",
        # pole4
    ]

    circles = [
        trilateration.Circle(trilateration.Point(0, 0),inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[0])['mean']),
        # trilateration.Circle(trilateration.Point(0, 0),inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[0])['median']),
        trilateration.Circle(trilateration.Point(50, 0),inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[1])['mean']),
        # trilateration.Circle(trilateration.Point(50, 0),inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[1])['median']),
        trilateration.Circle(trilateration.Point(0, 30),inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[2])['mean']),
        # trilateration.Circle(trilateration.Point(0, 30),inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[2])['median']),
        # trilateration.Circle(trilateration.Point(50, 30),inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[3])['mean']),
        # trilateration.Circle(trilateration.Porint(50, 30),inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[3])['median']),
    ]
    output_original = 0
    while True:
        output = trilateration.get_trilateration(circles)
        if output == "error":
            break
        elif output == 'ld':
            print('want long distance')
            for i in range(len(circles)):
                circles[i].radius = trilateration.increase_distance(circles[i].radius)
        else:
            output_original = (math.ceil(output.x), math.ceil(output.y))
            print('center x : ', math.ceil(output.x))
            print('center y : ', math.ceil(output.y))
            break

    # # ---- old fashion
    # # make the points in a 2d tuple if you want to use static points later
    # R1 = (0, 0)
    # R2 = (50, 0)
    # R3 = (50, 30)
    # # you have to introduce the distances
    # d1 = int(inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[0])['min'])
    # d2 = int(inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[1])['min'])
    # d3 = int(inference(nn_model=inference_model, model_config=model_config, file=pole_data_path[3])['min'])
    #
    # # if d1 ,d2 and d3 in known
    # # calculate A ,B and C coifficents
    # A = R1[0] ** 2 + R1[1] ** 2 - d1 ** 2
    # B = R2[0] ** 2 + R2[1] ** 2 - d2 ** 2
    # C = R3[0] ** 2 + R3[1] ** 2 - d3 ** 2
    # X32 = R3[0] - R2[0]
    # X13 = R1[0] - R3[0]
    # X21 = R2[0] - R1[0]
    #
    # Y32 = R3[1] - R2[1]
    # Y13 = R1[1] - R3[1]
    # Y21 = R2[1] - R1[1]
    #
    # x = (A * Y32 + B * Y13 + C * Y21) / (2.0 * (R1[0] * Y32 + R2[0] * Y13 + R3[0] * Y21))
    # y = (A * X32 + B * X13 + C * X21) / (2.0 * (R1[1] * X32 + R2[1] * X13 + R3[1] * X21))
    # # prompt the result
    # print("(x,y) = (" + str(x) + "," + str(y) + ")")
    print(output_original)

    start_time = time.time()
    print(time.time() - start_time)