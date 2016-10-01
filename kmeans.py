import sys
import math
import random
import numpy as np
import subprocess
from math import floor
from data_helper import load_data
import pickle

class Cluster:
    def __init__(self, points):
        if len(points) == 0:
            raise Exception("ILLEGAL: empty cluster")
        # The points that belong to this cluster
        self.points = points

        # The dimensionality of the points in this cluster
        self.n = points[0][0].size

        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculate_centroid()

    def update(self, points):
        old_centroid = np.array(self.centroid)
        self.points = points
        self.centroid = self.calculate_centroid()
        shift = np.linalg.norm(old_centroid - self.centroid)
        return shift

    def calculate_centroid(self):
        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*[i[0] for i in self.points])
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList) / numPoints for dList in unzipped]

        return np.array(centroid_coords)


def k_means(points, k, cutoff=0.5):
    # Pick out k random points to use as our initial centroids
    print 'Sampling random points'
    initial = random.sample(points, k)

    # Create k clusters using those centroids
    print 'Creating clusters'
    clusters = [Cluster([p]) for p in initial]

    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [ [] for c in clusters]
        clusterCount = len(clusters)

        # Start counting loops
        loopCounter += 1
        print '@ Iteration %i' % (loopCounter)
        # For every point in the dataset ...
        totalPoints = len(points)
        currentPoint = 0
        for p in points:
            currentPoint += 1

            percentage =  floor((float(currentPoint) / float(totalPoints)) * 100.)
            if currentPoint % (totalPoints // 5) == 0:
                print '%i%%' % (percentage)

            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = np.linalg.norm(p[0] - clusters[0].centroid)

            # Set the cluster this point belongs to
            clusterIndex = 0

            # For the remainder of the clusters ...
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = np.linalg.norm(p[0] - clusters[i+1].centroid)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is, and set the point to belong
                # to that cluster
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            lists[clusterIndex].append(p)

        # Get the biggest shift for a centroid.
        maxShift = 0.0

        print 'Updating clusters...'
        for i in range(clusterCount):
            shift = clusters[i].update(lists[i])
            maxShift = max(maxShift, shift)

        print 'Biggest shift of this iteration was ', maxShift

        # The convergence point for the centroids.
        if maxShift < cutoff:
            print "Converged after %s iterations" % loopCounter
            break

    return [clusters.centroid for cluster in clusters]

# def create_cluster():
#     num_points = 100
#
#     datasets = load_data('mnist.pkl', notShared=True)
#     trainSetX, trainSetY = datasets[0]
#
#     trainSet = zip(trainSetX, trainSetY)
#
#     nClusters = 10
#
#     # When do we say the optimization has 'converged' and stop updating clusters
#     convergenceCutoff = 0.5
#
#     clusters = kmeans(trainSet, nClusters, convergenceCutoff)
#
#     print 'Saving to file...'
#     with open('cluster_model.p', 'wb') as f:
#         centroids = [cluster.centroid for cluster in clusters]
#         pickle.dump(centroids, f)
#
#
# def load_centroids():
#     print 'Loading cluster data...'
#     with open('cluster_model.p', 'rb') as f:
#         centroids = pickle.load(f)
#         return centroids
#
#
# def calculate_score(centroids):
#     # Label the data.
#     print 'Getting correct label'
#
#     datasets = load_data('mnist.pkl', notShared=True)
#     testSetX, testSetY = datasets[2]
#     testSetPairs = zip(testSetX, testSetY)
#
#     nearestCentroids = {}
#
#     for testSetPair in testSetPairs:
#         # Find the nearest centroid
#         minDist = -1
#         minIndex = -1
#         for i, centroid in enumerate(centroids):
#             dist = np.linalg.norm(testSetPair[0] - centroid)
#
#             if minDist == -1 or dist < minDist:
#                 minDist = dist
#                 minIndex = i
#
#         if not nearestCentroids.has_key(minIndex):
#             nearestCentroids[minIndex] = [testSetPair[1]]
#         else:
#             nearestCentroids[minIndex].append(testSetPair[1])
#
#     # Find the distribution of the digit classes across the centroids.
#     nClasses = sum([len(nearestCentroids[k]) for k in nearestCentroids])
#     for centroidKey in nearestCentroids:
#         dists = {}
#         for i in range(10):
#             dists[i] = 0
#         for classScore in nearestCentroids[centroidKey]:
#             dists[classScore] += 1
#
#         print 'Breakdown for centroid ', centroidKey
#         maxDist = 0
#         maxIndex = -1
#         for k in dists:
#             dists[k] = (float(dists[k]) / float(nClasses)) * 100.0
#             if dists[k] > maxDist:
#                 maxDist = dists[k]
#                 maxIndex = k
#             print '     %i) %.1f%%' % (k, dists[k])
#         print 'Prominently %i' % (maxIndex)
