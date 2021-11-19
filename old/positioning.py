import torch
from model_pack import model
import numpy as np
from data import data_loader_normal
import random
import time
import math
from trilateration_booster_v1 import trilateration
# randomness
# random_seed = 11
# np.random.seed(random_seed)
# random.seed(random_seed)
# torch.manual_seed(random_seed)
# # 연산속도가 느려질 수 있음
# # torch.backends.cudnn.deterministic = True
# # torch.backends.cudnn.benchmark = False
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU


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
    "model_type": "Custom_CRNN", "input_size": 11, "sequence_length": 15, "activation": "ReLU",
    "convolution_layer": 1, "bidirectional": False, "hidden_size": 256, "num_layers": 1,
    "linear_layers": [64, 1], "criterion": "MSELoss", "optimizer": "AdamW", "dropout_rate": 0.5,
    "use_cuda": True, "batch_size": 200, "learning_rate": 0.0001, "epoch": 2000,
    "num_workers": 8, "shuffle": True, "input_dir": "dataset/v5/newb","cuda_num":'cuda:0',
    # "checkpoint_path": "test_checkpoints/quick06-4b_Custom_CRNN_AdamW_ReLU_0.0001_sl15_999_epoch_1279.pt"
    "checkpoint_path": "test_checkpoints/quick_01-01_Custom_CRNN_AdamW_ReLU_0.0001_sl15_999_epoch_1489.pt"
}

if __name__ == '__main__':
    start_time = time.time()

    # pol1_data_path = 'dataset/v5/test3_point_39_0_5/f9-9c-e9-5d-b9-6f_39_0_5_pol1.csv'
    # pol2_data_path = 'dataset/v5/test3_p
    # oint_39_0_5/f9-9c-e9-5d-b9-6f_39_0_5_pol2.csv'
    # pol3_data_path = 'dataset/v5/test3_point_39_0_5/f9-9c-e9-5d-b9-6f_39_0_5_pol3.csv'

    # pol1_data_path = 'dataset/v5/test3_point_39_5_15/04-ee-03-74-ae-a5_39_5_15_pol1.csv'
    # pol2_data_path = 'dataset/v5/test3_point_39_5_15/04-ee-03-74-ae-a5_39_5_15_pol2.csv'
    # pol3_data_path = 'dataset/v5/test3_point_39_5_15/04-ee-03-74-ae-a5_39_5_15_pol3.csv'

    # pol1_data_path = 'dataset/v5/test3_point_39_10_5/04-ee-03-74-ae-a5_39_10_5_pol1.csv'
    # pol2_data_path = 'dataset/v5/test3_point_39_10_5/04-ee-03-74-ae-a5_39_10_5_pol2.csv'
    # pol3_data_path = 'dataset/v5/test3_point_39_10_5/04-ee-03-74-ae-a5_39_10_5_pol3.csv'

    # pol1_data_path = 'dataset/v5/test3_point_39_10_5/04-ee-03-74-b0-30_39_10_5_pol1.csv'
    # pol2_data_path = 'dataset/v5/test3_point_39_10_5/04-ee-03-74-b0-30_39_10_5_pol2.csv'
    # pol3_data_path = 'dataset/v5/test3_point_39_10_5/04-ee-03-74-b0-30_39_10_5_pol3.csv'
    #
    # pol1_data_path = 'dataset/v5/test3_point_39_10_10/f9-9c-e9-5d-b9-6f_39_10_10_pol1.csv'
    # pol2_data_path = 'dataset/v5/test3_point_39_10_10/f9-9c-e9-5d-b9-6f_39_10_10_pol2.csv'
    # pol3_data_path = 'dataset/v5/test3_point_39_10_10/f9-9c-e9-5d-b9-6f_39_10_10_pol3.csv'

    # pol1_data_path = 'dataset/v5/test3_point_39_10_20/04-ee-03-74-ae-a5_39_10_20_pol1.csv'
    # pol2_data_path = 'dataset/v5/test3_point_39_10_20/04-ee-03-74-ae-a5_39_10_20_pol2.csv'
    # pol3_data_path = 'dataset/v5/test3_point_39_10_20/04-ee-03-74-ae-a5_39_10_20_pol3.csv'
    #
    # pol1_data_path = 'dataset/v5/test3_point_39_15_10/f9-9c-e9-5d-b9-6f_39_15_10_pol1.csv'
    # pol2_data_path = 'dataset/v5/test3_point_39_15_10/f9-9c-e9-5d-b9-6f_39_15_10_pol2.csv'
    # pol3_data_path = 'dataset/v5/test3_point_39_15_10/f9-9c-e9-5d-b9-6f_39_15_10_pol3.csv'

    # pol1_data_path = 'dataset/v5/test3_point_39_15_25/04-ee-03-74-ae-a5_39_15_25_pol1.csv'
    # pol2_data_path = 'dataset/v5/test3_point_39_15_25/04-ee-03-74-ae-a5_39_15_25_pol2.csv'
    # pol3_data_path = 'dataset/v5/test3_point_39_15_25/04-ee-03-74-ae-a5_39_15_25_pol3.csv'

    # pol1_data_path = 'dataset/v5/test3_point_39_20_15/04-ee-03-74-ae-a5_39_20_15_pol1.csv'
    # pol2_data_path = 'dataset/v5/test3_point_39_20_15/04-ee-03-74-ae-a5_39_20_15_pol2.csv'
    # pol3_data_path = 'dataset/v5/test3_point_39_20_15/04-ee-03-74-ae-a5_39_20_15_pol3.csv'
    #
    # pol1_data_path = 'dataset/v5/test3_point_39_20_15/04-ee-03-74-b0-30_39_20_15_pol1.csv'
    # pol2_data_path = 'dataset/v5/test3_point_39_20_15/04-ee-03-74-b0-30_39_20_15_pol2.csv'
    # pol3_data_path = 'dataset/v5/test3_point_39_20_15/04-ee-03-74-b0-30_39_20_15_pol3.csv'
    #
    ############
    #
    # pol1_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_37_0_5/04-ee-03-74-ae-a5_37_0_5_pol1.csv'
    # pol2_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_37_0_5/04-ee-03-74-ae-a5_37_0_5_pol2.csv'
    # pol3_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_37_0_5/04-ee-03-74-ae-a5_37_0_5_pol3.csv'

    # pol1_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_37_10_5/04-ee-03-74-ae-a5_37_10_5_pol1.csv'
    # pol2_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_37_10_5/04-ee-03-74-ae-a5_37_10_5_pol2.csv'
    # pol3_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_37_10_5/04-ee-03-74-ae-a5_37_10_5_pol3.csv'
    #
    # pol1_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_37_20_20/04-ee-03-74-ae-a5_37_20_20_pol1.csv'
    # pol2_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_37_20_20/04-ee-03-74-ae-a5_37_20_20_pol2.csv'
    # pol3_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_37_20_20/04-ee-03-74-ae-a5_37_20_20_pol3.csv'

    # pol1_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_39_5_10/04-ee-03-74-ae-a5_39_5_10_pol1.csv'
    # pol2_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_39_5_10/04-ee-03-74-ae-a5_39_5_10_pol2.csv'
    # pol3_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_39_5_10/04-ee-03-74-ae-a5_39_5_10_pol3.csv'

    # pol1_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_39_5_20/04-ee-03-74-ae-a5_39_5_20_pol1.csv'
    # pol2_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_39_5_20/04-ee-03-74-ae-a5_39_5_20_pol2.csv'
    # pol3_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_39_5_20/04-ee-03-74-ae-a5_39_5_20_pol3.csv'

    # pol1_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_39_10_15/04-ee-03-74-ae-a5_39_10_15_pol1.csv'
    # pol2_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_39_10_15/04-ee-03-74-ae-a5_39_10_15_pol2.csv'
    # pol3_data_path = 'dataset/v6/test1_point_04-ee-03-74-ae-a5_39_10_15/04-ee-03-74-ae-a5_39_10_15_pol3.csv'

    # pol1_data_path = 'dataset/v6/test2_point_04-ee-03-74-b0-30_37_10_5/04-ee-03-74-b0-30_37_10_5_pol1.csv'
    # pol2_data_path = 'dataset/v6/test2_point_04-ee-03-74-b0-30_37_10_5/04-ee-03-74-b0-30_37_10_5_pol2.csv'
    # pol3_data_path = 'dataset/v6/test2_point_04-ee-03-74-b0-30_37_10_5/04-ee-03-74-b0-30_37_10_5_pol3.csv'

    # pol1_data_path = 'dataset/v6/test2_point_04-ee-03-74-b0-30_39_5_10/04-ee-03-74-b0-30_39_5_10_pol1.csv'
    # pol2_data_path = 'dataset/v6/test2_point_04-ee-03-74-b0-30_39_5_10/04-ee-03-74-b0-30_39_5_10_pol2.csv'
    # pol3_data_path = 'dataset/v6/test2_point_04-ee-03-74-b0-30_39_5_10/04-ee-03-74-b0-30_39_5_10_pol3.csv'

    # ####### 5, 20 ########
    # pol1_data_path = 'dataset/v7/test1_point_f8-8a-5e-45-73-85_38_5_20/f8-8a-5e-45-73-85_38_5_20_pol1.csv'
    # pol2_data_path = 'dataset/v7/test1_point_f8-8a-5e-45-73-85_38_5_20/f8-8a-5e-45-73-85_38_5_20_pol2.csv'
    # # pol3_data_path = 'dataset/v7/test1_point_f8-8a-5e-45-73-85_38_5_20/f8-8a-5e-45-73-85_38_5_20_pol3.csv'
    #
    # inference_model = model.model_load(model_configure)
    #
    # circles = [trilateration.Circle(
    #                 trilateration.Point(0, 0),
    #                 inference(nn_model=inference_model, model_config=model_configure, file=pol1_data_path)['max']
    #             ), trilateration.Circle(
    #                 trilateration.Point(0, 50),
    #                 inference(nn_model=inference_model, model_config=model_configure,  file=pol2_data_path)['max']
    #             ),
    #             trilateration.Circle(
    #                 trilateration.Point(0, 0),
    #                 inference(nn_model=inference_model, model_config=model_configure, file=pol1_data_path)['min']
    #             ), trilateration.Circle(
    #                 trilateration.Point(0, 50),
    #                 inference(nn_model=inference_model, model_config=model_configure, file=pol2_data_path)['min']
    #             ),
    #             #     trilateration.Circle(
    #             #     trilateration.Point(30, 0),
    #             #     inference(nn_model=inference_model, model_config=model_configure,  file=pol3_data_path)['max'])
    #     ]
    #
    # ############

    # ####### 5, 25 ########
    # pol1_data_path = 'dataset/v7/test1_point_f8-8a-5e-45-6c-b6_37_5_25/f8-8a-5e-45-6c-b6_37_5_25_pol1.csv'
    # pol3_data_path = 'dataset/v7/test1_point_f8-8a-5e-45-6c-b6_37_5_25/f8-8a-5e-45-6c-b6_37_5_25_pol3.csv'
    # pol4_data_path = 'dataset/v7/test1_point_f8-8a-5e-45-6c-b6_37_5_25/f8-8a-5e-45-6c-b6_37_5_25_pol4.csv'
    #
    # inference_model = model.model_load(model_configure)
    #
    # circles = [trilateration.Circle(
    #     trilateration.Point(0, 0),
    #     inference(nn_model=inference_model, model_config=model_configure, file=pol1_data_path)['max']
    # ), trilateration.Circle(
    #     trilateration.Point(30, 0),
    #     inference(nn_model=inference_model, model_config=model_configure, file=pol3_data_path)['max']
    # ),
    #     trilateration.Circle(
    #         trilateration.Point(30, 50),
    #         inference(nn_model=inference_model, model_config=model_configure, file=pol4_data_path)['max']
    #     ),
    #     trilateration.Circle(
    #         trilateration.Point(0, 0),
    #         inference(nn_model=inference_model, model_config=model_configure, file=pol1_data_path)['median']
    #     ), trilateration.Circle(
    #         trilateration.Point(30, 0),
    #         inference(nn_model=inference_model, model_config=model_configure, file=pol3_data_path)['median']
    #     ),
    #     trilateration.Circle(
    #         trilateration.Point(30, 50),
    #         inference(nn_model=inference_model, model_config=model_configure, file=pol4_data_path)['median']
    #     ),
    #     #     trilateration.Circle(
    #     #     trilateration.Point(30, 0),
    #     #     inference(nn_model=inference_model, model_config=model_configure,  file=pol3_data_path)['max'])
    # ]
    #
    # ############

    #
    ####### 10, 25 ########
    pol1_data_path = '../dataset/v7/test1_point_f8-8a-5e-45-6c-b6_37_10_25/f8-8a-5e-45-6c-b6_37_10_25_pol1.csv'
    pol2_data_path = '../dataset/v7/test1_point_f8-8a-5e-45-6c-b6_37_10_25/f8-8a-5e-45-6c-b6_37_10_25_pol2.csv'
    pol3_data_path = '../dataset/v7/test1_point_f8-8a-5e-45-6c-b6_37_10_25/f8-8a-5e-45-6c-b6_37_10_25_pol3.csv'

    inference_model = model.model_load(model_configure)

    circles = [trilateration.Circle(
        trilateration.Point(0, 0),
        inference(nn_model=inference_model, model_config=model_configure, file=pol1_data_path)['min']
    ), trilateration.Circle(
        trilateration.Point(30, 0),
        inference(nn_model=inference_model, model_config=model_configure, file=pol2_data_path)['min']
    ),
        trilateration.Circle(
            trilateration.Point(0, 50),
            inference(nn_model=inference_model, model_config=model_configure, file=pol3_data_path)['min']
        ),
        # trilateration.Circle(
        #     trilateration.Point(0, 0),
        #     inference(nn_model=inference_model, model_config=model_configure, file=pol1_data_path)['median']
        # ), trilateration.Circle(
        #     trilateration.Point(50, 0),
        #     inference(nn_model=inference_model, model_config=model_configure, file=pol2_data_path)['median']
        # ),
        # trilateration.Circle(
        #     trilateration.Point(30, 0),
        #     inference(nn_model=inference_model, model_config=model_configure, file=pol3_data_path)['median']
        # ),
        #     trilateration.Circle(
        #     trilateration.Point(30, 0),
        #     inference(nn_model=inference_model, model_config=model_configure,  file=pol3_data_path)['max'])
    ]

    ############

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