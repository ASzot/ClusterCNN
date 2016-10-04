import numpy as np
import random
import pickle

class OnlineKMeans:
    def __init__(self, k, epsilon, startingData):
        self.centroids=[]
        self.nClusters = k
        self.eps = epsilon
        self.nearestClusterIndex=0
        self.dimension = -1
        self.centroids = random.sample(startingData, k)


    def process(self, point):
        point = np.array(point)

        if len(self.centroids) < self.nClusters:
            if self.dimension == -1:
                self.dimension = len(point)
            self.centroids.append(point)
            # returns the last cluster index
            self.nearestClusterIndex= len(self.centroids) - 1

        else:
            #indentifying nearest cluster
            self.nearestClusterIndex = self.get_nearest_cluster(point)

            #updating nearest cluster
            self.update_cluster(point)

        return self.nearestClusterIndex


    def update_cluster(self, point):
        nearestCluster = self.centroids[self.nearestClusterIndex]
        for i in range(self.dimension):
            self.centroids[self.nearestClusterIndex][i] = nearestCluster[i] + self.eps * (point[i] - nearestCluster[i])


    def get_nearest_cluster(self, point):
        min_distance = self.distance(point, 0)
        min_index = 0

        for clusterIndex in range(1, len(self.centroids)):
            tmp_dist = self.distance(point, clusterIndex)
            if tmp_dist < min_distance:
                min_distance = tmp_dist
                min_index = clusterIndex

        return min_index


    def distance(self, point, clusterIndex):
        return np.linalg.norm(point - self.centroids[clusterIndex])


def save_centroids(centroids, filename):
    print 'Saving to file...'
    with open(filename, 'wb') as f:
        pickle.dump(centroids, f)


def load_centroids(filename):
    print 'Loading cluster data...'
    with open(filename, 'rb') as f:
        centroids = pickle.load(f)
        return centroids
