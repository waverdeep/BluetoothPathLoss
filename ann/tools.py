import matplotlib.pyplot as plt


def plot_rssi_to_distance(ref_rssi, y_data, pred):
    plt.figure(figsize=(10, 10))
    plt.title('Validation RSSI to Distance', fontsize=20)
    plt.ylabel('Distance (meter)', fontsize=18)
    plt.xlabel('RSSI (dBm)', fontsize=18)
    plt.scatter(ref_rssi, y_data, c='b', label="groundtruth")
    plt.scatter(ref_rssi, pred, c='r', label="prediction")
    plt.legend()
    plt.grid(True)
    plt.show()