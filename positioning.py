import torch
from model import model
import numpy as np
from data import data_loader
import random
import time
import math
import trilateration_booster_v1.trilateration as trilateration
# randomness
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
# 연산속도가 느려질 수 있음
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU


def inference(model_config, file):
    nn_model = model.model_load(model_config)
    inference_dataloader = \
        data_loader.load_path_loss_with_detail_inference_dataset(file, 'CRNN',
                                                                 batch_size=model_config['batch_size'])
    total_pred = []
    with torch.no_grad():
        for i, data in enumerate(inference_dataloader):
            if model_config['model'] == 'DNN':
                pass
            elif model_config['model'] == 'RNN':
                pass
            elif model_config['model'] == 'CRNN':
                x_data = data.transpose(1, 2)
                x_data = x_data.cuda()
            y_pred = nn_model(x_data).reshape(-1).cpu().numpy()
            total_pred[len(total_pred):len(y_pred)] = y_pred

            print('mean : ', y_pred.mean())
            print('median : ', np.median(y_pred))
            print('max : ', y_pred.max())
            print('min : ', y_pred.min())
            return {'mean': y_pred.mean(), 'median': np.median(y_pred),
                    'max': y_pred.max(), 'min': y_pred.min()}


model_configure = {"model": "CRNN",
                   "activation": "LeakyReLU",
                   "cuda": True,
                   "batch_size": 512,
                   "input_size": 7,
                   "sequence_length": 15,
                   'checkpoint_path': 'checkpoints_all_v3/CRNN_Adam_LeakyReLU_0.001_sl15_000_epoch_769.pt',
}

if __name__ == '__main__':
    start_time = time.time()

    pol1_data_path = ''
    pol2_data_path = ''
    pol3_data_path = ''

    circles = [trilateration.Circle(
                    trilateration.Point(0, 0),
                    inference(model_config=model_configure, file=pol1_data_path)['max']
                ), trilateration.Circle(
                    trilateration.Point(30, 0),
                    inference(model_config=model_configure, file=pol2_data_path)['max']
                ), trilateration.Circle(
                    trilateration.Point(0, 30),
                    inference(model_config=model_configure, file=pol3_data_path)['max']
    )]

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