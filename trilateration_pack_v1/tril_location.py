import math


def get_distance_with_rssi(rssi, tx_power, free_space=2):
    return math.pow(10, (tx_power - rssi) / (10 * free_space))


class Point:
    def __init__(self, x, y, distance=0):
        self.x = x
        self.y = y
        self.distance = distance

    def get_distance(self, other_point, overlap=True):
        temp = math.sqrt(math.pow(self.x - other_point.x, 2) + math.pow(self.y - other_point.y, 2))
        if overlap:
            self.distance = temp
        return temp


class TrilaterationInCoord:
    def __init__(self, point1, point2, point3):
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3

    def get_trilateration(self):
        s = (math.pow(self.point3.x, 2) - math.pow(self.point2.x, 2)
             + math.pow(self.point3.y, 2) - math.pow(self.point2.y, 2)
             + math.pow(self.point2.distance, 2) - math.pow(self.point3.distance, 2)) / 2

        t = (math.pow(self.point1.x, 2) - math.pow(self.point2.x, 2)
             + math.pow(self.point1.y, 2) - math.pow(self.point2.y, 2)
             + math.pow(self.point2.distance, 2) - math.pow(self.point1.distance, 2)) / 2

        y = ((t*(self.point2.x - self.point3.x)) -
             (s*(self.point2.x - self.point1.x))) / (((self.point1.y - self.point2.y)*(self.point2.x - self.point3.x)) -
                                                     ((self.point3.y - self.point2.y)*(self.point2.x - self.point1.x)))

        x = ((y*(self.point1.y - self.point2.y)) - t)/(self.point2.x - self.point1.x)

        location = Point(x, y, 0)
        return location