import math
from itertools import combinations
import time
import model_pack.model_pathloss as model_pathloss
from old import positioning_v8 as positioning
import model_pack.model_reconst as model_reconst


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_distance(self, other_point):
        temp = math.sqrt(math.pow(self.x - other_point.x, 2) + math.pow(self.y - other_point.y, 2))
        return temp

    def __str__(self):
        print("[{}, {}]".format(self.x, self.y))


class Circle:
    def __init__(self, point, radius):
        self.point = point
        self.radius = radius

    # 두 원이 만나는 점이 있는지
    def get_intersecting_points(self, other_circle):
        distance = self.point.get_distance(other_circle.point)
        if distance >= abs(self.radius + other_circle.radius) or distance <= abs(self.radius - other_circle.radius):
            return None

        a = (pow(self.radius, 2) - pow(other_circle.radius, 2) + pow(distance, 2)) / (2*distance)
        h = math.sqrt(pow(self.radius, 2) - pow(a, 2))
        x0 = self.point.x + a * (other_circle.point.x - self.point.x) / distance
        y0 = self.point.y + a * (other_circle.point.y - self.point.y) / distance
        rx = -(other_circle.point.y - self.point.y) * (h/distance)
        ry = -(other_circle.point.x - self.point.x) * (h/distance)
        return [Point(x0 + rx, y0 - ry), Point(x0 - rx, y0 + ry)]


def get_intersecting_all_points(other_circles):
    dataset = list(combinations(other_circles, 2))
    points = []
    for data in dataset:
        output = data[0].get_intersecting_points(data[1])
        if output:
            points.extend(output)
    return points


def is_contained_in_circles(point, circles):
    for circle in circles:
        if point.get_distance(circle.point) > circle.radius:
            return False
    return True


def get_center(points):
    center = Point(0, 0)
    for point in points:
        center.x += point.x
        center.y += point.y

    center.x /= len(points)
    center.y /= len(points)
    return center


def get_trilateration(circles):
    size = len(circles)
    inner_points = []
    if size <= 2:
        print('cannot calculate')
    elif size >= 3:
        for point in get_intersecting_all_points(circles):
            if is_contained_in_circles(point, circles):
                inner_points.append(point)
        if len(inner_points) == 0:
            return "ld"
        for i in inner_points:
            print("[{},{}] ".format(i.x, i.y))
        center = get_center(inner_points)
        return center
    return "error"


def increase_distance(distance):
    rssi = model_pathloss.fspl_model_inverse(distance)
    rssi = rssi - 1
    new_distance = model_pathloss.fspl_model(rssi)
    return new_distance


model_configure = {
    "model_type": "DilatedCRNNSmallV8", "input_size": 3, "sequence_length": 15, "activation": "ReLU",
    "convolution_layer": 4, "bidirectional": False, "hidden_size": 64, "num_layers": 1, "linear_layers": [64, 1],
    "criterion": "MSELoss", "optimizer": "SGD", "dropout_rate": 0.5, "use_cuda": True, "cuda_num": "cuda:0",
    "batch_size": 32, "learning_rate": 0.01, "epoch": 800, "num_workers": 8, "shuffle": True,
    "input_dir": "dataset/v8/type03_train", "tensorboard_writer_path": "runs_2021_11_01",
    "section_message": "Type03-ReLU-sgd0.01-T71", "checkpoint_dir": "checkpoints/2021_11_01",
    # "checkpoint_path": "../test_checkpoints/Type02-ReLU-sgd0.0025-1024-2-T26_epoch_799.pt" # type02
    # "checkpoint_path": "../checkpoints/2021_11_03/Type03-ReLU-T04_epoch_537.pt"
    # "checkpoint_path": "../test_checkpoints/Type01-ReLU-sgd0.0025-T33_epoch_362.pt" # type01
    # "checkpoint_path": "../checkpoints/2021_11_04/Type01-ed-PReLU-512-128-smallv1-T001_epoch_799.pt" # type01
    "checkpoint_path": "../checkpoints/2021_11_04/Type01-smallv8-adamw0.0005-T001+weak-SGD0.001_epoch_300.pt" # type01
}


if __name__ == '__main__':
    start_time = time.time()
    inference_model = model_reconst.model_load(model_configure)
    # pole01 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-77-06_39_40_0/f8-8a-5e-45-77-06_39_40_0_pol0-0.csv"
    # pole02 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-77-06_39_40_0/f8-8a-5e-45-77-06_39_40_0_pol50-0.csv"
    # pole03 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-77-06_39_40_0/f8-8a-5e-45-77-06_39_40_0_pol0-30.csv"
    # pole04 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-71-85_37_35_20/f8-8a-5e-45-71-85_37_35_20_pol50-30.csv"

    # pole01 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-6c-c1_37_25_15/f8-8a-5e-45-6c-c1_37_25_15_pol0-0.csv"
    # pole03 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-6c-c1_37_25_15/f8-8a-5e-45-6c-c1_37_25_15_pol0-30.csv"

    # pole01 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-71-80_37_50_15/f8-8a-5e-45-71-80_37_50_15_pol0-0.csv"
    # pole02 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-71-80_37_50_15/f8-8a-5e-45-71-80_37_50_15_pol50-0.csv"
    # # pole03 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-71-80_37_50_15/f8-8a-5e-45-71-80_37_50_15_pol50-0.csv"
    # pole04 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-71-80_37_50_15/f8-8a-5e-45-71-80_37_50_15_pol50-30.csv"

    # pole01 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-71-80_38_5_20/f8-8a-5e-45-71-80_38_5_20_pol0-0.csv"
    # pole03 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-71-80_38_5_20/f8-8a-5e-45-71-80_38_5_20_pol0-30.csv"

    pole01 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-71-85_37_35_20/f8-8a-5e-45-71-85_37_35_20_pol0-0.csv"
    pole02 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-71-85_37_35_20/f8-8a-5e-45-71-85_37_35_20_pol50-0.csv"
    pole04 = "../dataset/v8/test_dataset01/test1_type01_point_f8-8a-5e-45-71-85_37_35_20/f8-8a-5e-45-71-85_37_35_20_pol50-30.csv"

    circles = []
    # min_data01 = positioning.inference(inference_model, model_configure, pole01)['min']
    # if min_data01 < 0:
    #     min_data01 = abs(min_data01)%10
    # circles.append(Circle(Point(0, 0), min_data01))
    # circles.append(Circle(Point(0, 0), positioning.inference(inference_model, model_configure, pole01)['max']))
    circles.append(Circle(Point(0, 0), positioning.inference(inference_model, model_configure, pole01)['mean']))
    circles.append(Circle(Point(0, 0), positioning.inference(inference_model, model_configure, pole01)['median']))
    print('----')
    # min_data02 = positioning.inference(inference_model, model_configure, pole02)['min']
    # if min_data02 < 0:
    #     min_data02 = abs(min_data02)%10
    # circles.append(Circle(Point(50, 0), min_data02))
    # circles.append(Circle(Point(50, 0), positioning.inference(inference_model, model_configure, pole02)['max']))
    circles.append(Circle(Point(50, 0), positioning.inference(inference_model, model_configure, pole02)['mean']))
    circles.append(Circle(Point(50, 0), positioning.inference(inference_model, model_configure, pole02)['median']))
    print('----')
    # min_data03 = positioning.inference(inference_model, model_configure, pole03)['min']
    # if min_data03 < 0:
    #     min_data03 = abs(min_data03)%10
    # circles.append(Circle(Point(0, 30), min_data03))
    # circles.append(Circle(Point(0, 30), positioning.inference(inference_model, model_configure, pole03)['max']))
    # circles.append(Circle(Point(0, 30), positioning.inference(inference_model, model_configure, pole03)['mean']))
    # circles.append(Circle(Point(0, 30), positioning.inference(inference_model, model_configure, pole03)['median']))
    print('----')
    # min_data04 = positioning.inference(inference_model, model_configure, pole04)['min']
    # if min_data04 < 0:
    #     min_data04 = abs(min_data04)%10
    # circles.append(Circle(Point(50, 30), min_data04))
    # circles.append(Circle(Point(50, 30), positioning.inference(inference_model, model_configure, pole04)['max']))
    circles.append(Circle(Point(50, 30), positioning.inference(inference_model, model_configure, pole04)['mean']))
    circles.append(Circle(Point(50, 30), positioning.inference(inference_model, model_configure, pole04)['median']))
    print('----')



    flag = True
    while flag:
        output = get_trilateration(circles)
        if output == "error":
            break
        elif output == 'ld':
            print('want long distance')
            for i in range(len(circles)):
                circles[i].radius = increase_distance(circles[i].radius)
        else:
            print('center x : ', math.ceil(output.x))
            print('center y : ', math.ceil(output.y))
            break

    print(time.time() - start_time)

