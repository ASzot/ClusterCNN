from sklearn.cluster import MiniBatchKMeans, KMeans
import pickle
import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise
import sklearn.preprocessing as preprocessing
from clustering_cosine import cosine_kmeans
import csv

def kmeans(input_data, k, batch_size):
    return cosine_kmeans(input_data, k)


def get_image_patches(inputImg, inputShape, stride, filterShape):
    # Reconstruct the image as a matrix.
    # imageMat = inputImg.reshape(inputShape[0], inputShape[1], inputShape[2])

    # Get the patch.
    rowOffset = 0
    colOffset = 0
    patches = []

    # Remember the receptive field acts across the entire depth parameter.
    while rowOffset < inputShape[1] - filterShape[0]:
        while colOffset < inputShape[2] - filterShape[1]:
            patch = [filterMat[rowOffset:rowOffset+filterShape[0], colOffset:colOffset+filterShape[1]] for filterMat in inputImg]
            patch = np.array(patch)
            patches.append(patch)
            colOffset += stride[1]
        rowOffset += stride[0]
        colOffset = 0

    return patches

def build_patch_vecs(dataSetX, inputShape, stride, filterShape):
    patchVecs = []
    total = len(dataSetX)
    for i, dataX in enumerate(dataSetX):
        # Print every percent
        if i % (total // 100) == 0:
            print '%.2f%%' % ((float(i) / float(total)) * 100.)
        patches = get_image_patches(dataX, inputShape, stride, filterShape)

        # Flatten each of the vectors.
        addPatchVecs = [patch.reshape(patch.shape[0] * patch.shape[1] * patch.shape[2]) for patch in patches]
        patchVecs.extend(addPatchVecs)

    return patchVecs


def save_centroids(centroids, filename):
    print 'Saving to file...'
    with open(filename, 'wb') as f:
        pickle.dump(centroids, f)


def load_centroids(filename):
    print 'Loading cluster data...'
    with open(filename, 'rb') as f:
        centroids = pickle.load(f)
        return np.array(centroids)

def construct_centroids(batch_size, trainSetX, input_shape, stride, filter_shape, k, convolute):
    print '- Building centroids'
    if convolute:

        clusterVecs = build_patch_vecs(trainSetX, input_shape, stride, filter_shape)

        # Save the cluster vectors.
        with open('data/kmeansdata/input.csv', 'wb') as f:
            data_writer = csv.writer(f, delimiter=',')
            for cluster_vec in clusterVecs:
                data_writer.write(cluster_vec)
        raise ValueError()
    else:
        # Flatten the input.
        sp = trainSetX.shape

        # Not garunteed to be 3 dimensions as the input will be flattened.
        # This is different than performing the convolution where it has to be 3 dimensional.
        input_shape_prod = 1.0
        for input_shape_dim in input_shape:
            input_shape_prod = input_shape_prod * input_shape_dim
        clusterVecs = trainSetX.reshape(sp[0], input_shape_prod)

    clusterVecs = np.array(clusterVecs)

    print 'Normalizing'
    clusterVecs = preprocessing.normalize(clusterVecs, norm='l2')

    print 'Beginning k - menas'
    centroids = kmeans(clusterVecs, k, batch_size)

    if convolute:
        # Expand the output.
        sp = centroids.shape
        centroids = centroids.reshape(sp[0], input_shape[0], filter_shape[0], filter_shape[1])
    return centroids


def load_or_create_centroids(forceCreate, filename, batch_size, dataSetX, input_shape, stride, filter_shape, k, target_std, convolute=True):
    if not forceCreate:
        try:
            centroids = load_centroids(filename)
        except IOError:
            forceCreate = True

    if forceCreate:
        centroids = construct_centroids(batch_size, dataSetX, input_shape, stride, filter_shape, k, convolute)
        save_centroids(centroids, filename)

    # Scale the centroids by some factor.
    # Apply normalization process to the centroids.
    if target_std is not None:
        # Scale the data so that it has the same std.
        std_method = 'mag'
        if std_method == 'mag':
            print 'doing mag scaling'
            mags = [np.linalg.norm(centroid) for centroid in centroids]
            current_std = np.std(mags)
        elif std_method == 'angle':
            pass
        elif std_method == 'raw':
            current_std = np.std(centroids)
        else:
            raise ValueError()

        std_scaling_fact = target_std / current_std
        print std_scaling_fact

        # centroids = [centroid / np.linalg.norm(centroid) for centroid in centroids]
        centroids = [centroid * std_scaling_fact for centroid in centroids]

        # centroids = [centroid * std_scaling_fact for centroid in centroids]

        centroids = np.array(centroids)

    return centroids
