import torch
from model_pack import model
import numpy as np
from data import data_loader_normal
import time
import math
from trilateration_booster_v1 import trilateration


def inference(nn_model, model_config, file):
    nn_model = nn_model
    inference_dataloader = \
        data_loader.load_path_loss_with_detail_inference_dataset(file, 'CRNN',
                                                                 batch_size=model_config['batch_size'])
    total_pred = []
    with torch.no_grad():
        for i, data in enumerate(inference_dataloader):
            print(data.shape)
            if 'model' in model_config:
                if model_config['model'] == 'DNN':
                    pass
                elif model_config['model'] == 'RNN':
                    pass
                elif model_config['model'] == 'CRNN':
                    x_data = data.transpose(1, 2)
                    x_data = x_data.cuda()
            elif 'model_type' in model_config:
                if model_config['model_type'] == 'Custom_DNN':
                    pass
                elif model_config['model_type'] == 'Custom_RNN':
                    pass
                elif model_config['model_type'] == 'Custom_CRNN':
                    x_data = data.transpose(1, 2)
                    x_data = x_data.cuda()
            y_pred = nn_model(x_data).reshape(-1).cpu().numpy()
            total_pred.extend(y_pred)
            # total_pred[len(total_pred):len(y_pred)] = y_pred

        # print('mean : ', y_pred.mean())
        # print('median : ', np.median(y_pred))
        # print('max : ', y_pred.max())
        # print('min : ', y_pred.min())
        # return {'mean': y_pred.mean(), 'median': np.median(y_pred),
        #         'max': y_pred.max(), 'min': y_pred.min()}
        total_pred = np.array(total_pred)
        print('mean : ', total_pred.mean())
        print('median : ', np.median(total_pred))
        print('max : ', total_pred.max())
        print('min : ', total_pred.min())
        return {'mean': total_pred.mean(), 'median': np.median(total_pred),
                'max': total_pred.max(), 'min': total_pred.min()}


model_configure = {
    "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "LeakyReLU",
            "convolution_layer": 2, "bidirectional": False, "hidden_size": 256, "num_layers": 1,
            "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
            "use_cuda": True, "batch_size": 8000, "learning_rate": 0.0005, "epoch": 7000,
            "num_workers": 8, "shuffle": True, "input_dir": "dataset/v7_all/point_all_v6","cuda_num":'cuda:0',
    # "checkpoint_path": "test_checkpoints/quick06-4b_Custom_CRNN_AdamW_ReLU_0.0001_sl15_999_epoch_1279.pt"
    "checkpoint_path": "test_checkpoints/quick_01-09_Custom_CRNN_AdamW_LeakyReLU_0.0005_sl15_999_epoch_6999.pt"
}

if __name__ == '__main__':
    start_time = time.time()



    # ####### 45, 40 ########
    pol5_data_path = '../dataset/v7_all/test1_point_c0-be-b0-c3-d6-3d_37_5_40/c0-be-b0-c3-d6-3d_37_5_40_pol0-60.csv'
    pol6_data_path = '../dataset/v7_all/test1_point_c0-be-b0-c3-d6-3d_37_5_40/c0-be-b0-c3-d6-3d_37_5_40_pol50-60.csv'
    # pol6_data_path = 'dataset/v7_all/test1_point_f8-8a-5e-45-6c-b6_37_25_25/f8-8a-5e-45-6c-b6_37_25_25_pol50-60.csv'
    # pol6_data_path = 'dataset/v7_all/test1_point_04-ee-03-74-b0-30_38_30_30/04-ee-03-74-b0-30_38_30_30_pol50-60.csv'



    inference_model = model.model_load(model_configure)

    # print(inference(nn_model=inference_model, model_config=model_configure, file=pol5_data_path)['min'])

    circles = [
        trilateration.Circle(
            trilateration.Point(0, 60),
            inference(nn_model=inference_model, model_config=model_configure, file=pol5_data_path)['min']
        ),
        trilateration.Circle(
            trilateration.Point(50, 60),
            inference(nn_model=inference_model, model_config=model_configure,  file=pol6_data_path)['min']
        ),
        trilateration.Circle(
            trilateration.Point(0, 60),
            inference(nn_model=inference_model, model_config=model_configure, file=pol5_data_path)['max']
        ),
        trilateration.Circle(
            trilateration.Point(50, 60),
            inference(nn_model=inference_model, model_config=model_configure, file=pol6_data_path)['max']
        ),
    ]

    # ############

    while True:
        output = trilateration.get_trilateration(circles)
        if output == "error":
            break
        elif output == 'ld':
            print('want long distance')
            for i in range(len(circles)):
                circles[i].radius = trilateration.increase_distance(circles[i].radius)
        else:
            print('center x : ', math.ceil(output.x))
            print('center y : ', math.ceil(output.y))
            break

    print(time.time() - start_time)