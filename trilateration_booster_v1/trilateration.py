import math
from itertools import combinations
import time
import model.model_pathloss as model_pathloss


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_distance(self, other_point):
        temp = math.sqrt(math.pow(self.x - other_point.x, 2) + math.pow(self.y - other_point.y, 2))
        return temp


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
        center = get_center(inner_points)
        return center
    return "error"


def increase_distance(distance):
    rssi = model_pathloss.fspl_model_inverse(distance)
    rssi = rssi - 1
    new_distance = model_pathloss.fspl_model(rssi)
    return new_distance


model_configure = {"model": "CRNN", "criterion": "MSELoss","optimizer": "Adam","activation": "LeakyReLU",
                   "learning_rate": 0.001, "cuda": True, "batch_size": 512, "epoch": 800, "input_size": 11,
                   "sequence_length": 15, "input_dir": "dataset/v1_scaled_mm", "shuffle": True, "num_workers": 8,
                   'checkpoint_path': '../checkpoints/v5_v1/CRNN_Adam_LeakyReLU_0.001_sl15_000_epoch_569.pt',
                   'test_data_path': 'dataset/v3_test_convert', 'scaler_path': 'data/MinMaxScaler_saved.pkl'}
# ../checkpoints_all_v3/CRNN_Adam_LeakyReLU_0.001_sl15_000_epoch_769.pt
# ../checkpoints_all_v3_re/CRNN_Adam_LeakyReLU_0.001_sl15_000_epoch_829.pt
# ../checkpoints_all_v4/CRNN_Adam_LeakyReLU_0.001_sl15_000_epoch_879.pt
# ../checkpoints/v5_v1/CRNN_Adam_LeakyReLU_0.001_sl15_000_epoch_569.pt

if __name__ == '__main__':
    start_time = time.time()
    circles = []
    # circles.append(Circle(Point(0, 0), positioning.inference(model_configure,
    #                                                               '../dataset/v5/test_point_00_15/fb-f8-c9-9d-d8-d6_37_0_15_pol1.csv')['median']))
    # circles.append(Circle(Point(30, 0), positioning.inference(model_configure,
    #                                                                '../dataset/v5/test_point_00_15/fb-f8-c9-9d-d8-d6_37_0_15_pol2.csv')['median']))
    # circles.append(Circle(Point(0, 30), positioning.inference(model_configure,
    #                                                                '../dataset/v5/test_point_00_15/fb-f8-c9-9d-d8-d6_37_0_15_pol3.csv')['median']))

    circles.append(Circle(Point(0, 0), 15))
    circles.append(Circle(Point(30, 0), 15))
    circles.append(Circle(Point(0, 30), 30.577108))

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

