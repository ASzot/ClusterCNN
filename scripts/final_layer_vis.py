import pickle
from sklearn.manifold import TSNE
import csv

import numpy as np

def main():
    centroids = []
    with open('data/centroids/python_kmeans/cluster/f2.csv') as f:
        reader = csv.reader(f)
        for centroid in reader:
            centroids.append(centroid)

    centroids = np.array(centroids)

    print centroids.shape



if __name__ == '__main__':
    main()
