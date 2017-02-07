from tsne import bh_sne
from matplotlib import pyplot as plt
import csv
import numpy as np


def main():
    centroids = []
    with open('data/centroids/python_kmeans/cluster/f2.csv', 'rb') as f:
        reader = csv.reader(f)
        for centroid in reader:
            centroids.append(centroid)
    centroids = np.array(centroids)
    print centroids[0]
    return
    vis_data = bh_sne(centroids)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.show()


if __name__ == "__main__":
    main()

