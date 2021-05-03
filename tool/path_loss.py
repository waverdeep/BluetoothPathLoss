import math
import numpy as np


def get_distance_with_rssi_fspl(rssi, freq=2412 * math.pow(10, 6)):
    return math.pow(10, (-rssi - 20 * math.log10(freq) + 147.55) / 20)


def get_distance_with_rssi(rssi, tx_power, free_space=2):
    return math.pow(10, (tx_power - rssi) / (10 * free_space))


def log_distance(rssi):
    freq = 2412 * math.pow(10, 6)  # Transmission frequency
    txPower = 5  # Transmission power in dB
    antennaeGain = 0  # Total antennae gains (transmitter + receiver) in dB
    refDist = 1  # Reference distance from the transceiver in meters
    lightVel = 3e8  # Speed of light
    refLoss = 20 * np.log10(
        (4 * np.pi * refDist * freq / lightVel))  # Free space path loss at the reference distance (refDist)
    ERP = txPower + antennaeGain - refLoss  # Kind of the an equivalent radiation power
    return math.pow(10, ((rssi - ERP + 17.76) / (-10 * 1.92)))