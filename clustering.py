from sklearn.cluster import MiniBatchKMeans, KMeans
import pickle
import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import pairwise
import sklearn.preprocessing as preprocessing
from clustering_cosine import cosine_kmeans
from scipy.cluster.vq import whiten
import csv

def kmeans(input_data, k, batch_size, metric='euclidean'):
    if (metric == 'cosine'):
        print 'Normalizing'
        input_data = preprocessing.normalize(input_data, norm='l2')
        return cosine_kmeans(input_data, k)
    elif metric == 'euclidean':
        mbk = MiniBatchKMeans(init='k-means++',
                                n_clusters=k,
                                batch_size=batch_size,
                                max_no_improvement=10,
                                reassignment_ratio=0.01,
                                verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mbk.fit(input_data)
        return mbk.cluster_centers_
    else:
        raise NameError()


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
        patches = get_image_patches(dataX, inputShape, stride, filterShape)

        # Flatten each of the vectors.
        addPatchVecs = [patch.reshape(patch.shape[0] * patch.shape[1] * patch.shape[2]) for patch in patches]
        patchVecs.extend(addPatchVecs)

    return patchVecs


def save_centroids(centroids, filename):
    print 'Saving to file...'
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for centroid in centroids:
            writer.writerow(centroid)


def load_centroids(filename):
    print 'Loading cluster data...'
    centroids = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for centroid in reader:
            centroids.append(centroid)
    return np.array(centroids)


def construct_centroids(raw_save_loc, batch_size, train_set_x, input_shape, stride, filter_shape, k, convolute):
    print '- Building centroids'
    if convolute:
        cluster_vecs = build_patch_vecs(train_set_x, input_shape, stride, filter_shape)
    else:
        # Flatten the input.
        sp = train_set_x.shape

        # Not garunteed to be 3 dimensions as the input will be flattened.
        # This is different than performing the convolution where it has to be 3 dimensional.
        input_shape_prod = 1.0
        for input_shape_dim in input_shape:
            input_shape_prod = input_shape_prod * input_shape_dim
        cluster_vecs = train_set_x.reshape(sp[0], int(input_shape_prod))

    cluster_vecs = np.array(cluster_vecs)

    print 'Whitening data.'
    cluster_vecs = whiten(cluster_vecs)

    print 'Mean centering'
    cluster_vec_mean = np.mean(cluster_vecs)
    cluster_vecs = cluster_vecs - cluster_vec_mean

    if raw_save_loc != '':
        print 'Saving image patches'
        with open(raw_save_loc, 'wb') as f:
            csvwriter = csv.writer(f)
            for cluster_vec in cluster_vecs:
                csvwriter.writerow(cluster_vec)

    print 'Beginning k - means'
    centroids = kmeans(cluster_vecs, k, batch_size)

    # Normalize each of the centroids.
    for i, centroid in enumerate(centroids):
        centroids[i] = (centroid / np.linalg.norm(centroid))

    return centroids


def load_or_create_centroids(forceCreate, filename, batch_size, dataSetX, input_shape, stride, filter_shape, k, convolute=True, scale_factor = 1.0, raw_save_loc=''):
    if not forceCreate:
        try:
            centroids = load_centroids(filename)
        except IOError:
            forceCreate = True

    if forceCreate:
        centroids = construct_centroids(raw_save_loc, batch_size, dataSetX, input_shape, stride, filter_shape, k, convolute)
        # Whiten the data.
        # centroids = whiten(centroids)

        save_centroids(centroids, filename)

    return centroids
