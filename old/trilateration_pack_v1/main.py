import tril_location
import math
import matplotlib.pyplot as plt


def prev_v1():
    point1 = tril_location.Point(0, 0, math.sqrt(50))
    point2 = tril_location.Point(10, 0, math.sqrt(50))
    point3 = tril_location.Point(5, 10, 5)

    tril = tril_location.TrilaterationInCoord(point1, point2, point3)
    location = tril.get_trilateration()
    print("x : {}".format(location.x))
    print("y : {}".format(location.y))

    print("-------------------------------------------")

    point_A = tril_location.Point(0, 0)
    point_B = tril_location.Point(22, 0)
    point_C = tril_location.Point(0, 50)

    point_m1 = tril_location.Point(14, 7)
    point_m2 = tril_location.Point(12, 13)
    point_m3 = tril_location.Point(6, 20)
    point_m4 = tril_location.Point(6, 27)

    m1_to_a_distance = point_A.get_distance(point_m1)
    m1_to_b_distance = point_B.get_distance(point_m1)
    m1_to_c_distance = point_C.get_distance(point_m1)

    print('m1 to A distance : {}'.format(m1_to_a_distance))
    print('m1 to B distance : {}'.format(m1_to_b_distance))
    print('m1 to C distance : {}'.format(m1_to_c_distance))

    tril = tril_location.TrilaterationInCoord(point_A, point_B, point_C)
    location = tril.get_trilateration()
    print("x : {}".format(location.x))
    print("y : {}".format(location.y))

    print("-------------------------------------------")

    m2_to_a_distance = point_A.get_distance(point_m2)
    m2_to_b_distance = point_B.get_distance(point_m2)
    m2_to_c_distance = point_C.get_distance(point_m2)

    print('m2 to A distance : {}'.format(m2_to_a_distance))
    print('m2 to B distance : {}'.format(m2_to_b_distance))
    print('m2 to C distance : {}'.format(m2_to_c_distance))

    tril = tril_location.TrilaterationInCoord(point_A, point_B, point_C)
    location = tril.get_trilateration()
    print("x : {}".format(location.x))
    print("y : {}".format(location.y))

    print("-------------------------------------------")

    m3_to_a_distance = point_A.get_distance(point_m3)
    m3_to_b_distance = point_B.get_distance(point_m3)
    m3_to_c_distance = point_C.get_distance(point_m3)

    print('m3 to A distance : {}'.format(m3_to_a_distance))
    print('m3 to B distance : {}'.format(m3_to_b_distance))
    print('m3 to C distance : {}'.format(m3_to_c_distance))

    tril = tril_location.TrilaterationInCoord(point_A, point_B, point_C)
    location = tril.get_trilateration()
    print("x : {}".format(location.x))
    print("y : {}".format(location.y))

    print("-------------------------------------------")

    m4_to_a_distance = point_A.get_distance(point_m4)
    m4_to_b_distance = point_B.get_distance(point_m4)
    m4_to_c_distance = point_C.get_distance(point_m4)

    print('m4 to A distance : {}'.format(m4_to_a_distance))
    print('m4 to B distance : {}'.format(m4_to_b_distance))
    print('m4 to C distance : {}'.format(m4_to_c_distance))

    tril = tril_location.TrilaterationInCoord(point_A, point_B, point_C)
    location = tril.get_trilateration()
    print("x : {}".format(location.x))
    print("y : {}".format(location.y))

    print("-------------------------------------------")

    point_A = tril_location.Point(0, 0, 18.4)
    point_B = tril_location.Point(22, 0, 12.1)
    point_C = tril_location.Point(0, 50, 34.16)

    tril = tril_location.TrilaterationInCoord(point_A, point_B, point_C)
    location = tril.get_trilateration()
    print("x : {}".format(location.x))
    print("y : {}".format(location.y))

    print("-------------------------------------------")

    point_m1_pred = tril_location.Point(1.6, 18.97)
    point_m3_pred = tril_location.Point(15.36, 16.7)
    point_m4_pred = tril_location.Point(24.59, 22.6)

    print(point_m4_pred.get_distance(point_m1))
    print(point_m4_pred.get_distance(point_m3))
    print(point_m4_pred.get_distance(point_m4))


def show_position(ball, pred):
    plt.scatter(ball.x, ball.y, c='r')
    plt.scatter(pred.x, pred.y, c='b')
    plt.show()


if __name__ == '__main__':
    ball = tril_location.Point(4, 3)
    pol_a = tril_location.Point(0, 0)
    print('pol_a distance : ', pol_a.get_distance(ball))
    pol_b = tril_location.Point(10, 0)
    print('pol_b distance : ', pol_b.get_distance(ball))
    pol_c = tril_location.Point(10, 9)
    print('pol_c distance : ', pol_c.get_distance(ball))

    # pol_a.distance  -= 1

    tril = tril_location.TrilaterationInCoord(pol_a, pol_b, pol_c)
    output = tril.get_trilateration()
    print('---ball position---')
    output.out()

    show_position(ball, output)










