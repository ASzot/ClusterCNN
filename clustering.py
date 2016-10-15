from sklearn.cluster import MiniBatchKMeans, KMeans
import pickle
import numpy as np
import warnings
from sklearn.spatial.distance import cosine_distance


def new_euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
    # Use the cosine distance instead of the eulicidean distance.
    return cosine_distance(X, Y)

# This should work for k means mini batch as well.
from sklearn.cluster import k_means_k_means_.euclidean_distances
k_means_.euclidean_distances = new_euclidean_distances

def kmeans(inputData, k, batch_size):
    mbk = MiniBatchKMeans(init='k-means++',
                        n_clusters=k,
                        batch_size=batch_size,
                        verbose=False)

    # mbk = KMeans(init='k-means++', n_clusters=k, verbose=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mbk.fit(inputData)

    return mbk.cluster_centers_


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
            patches.append(np.array(patch))
            colOffset += stride[1]
        rowOffset += stride[0]
        colOffset = 0

    return patches

def build_patch_vecs(dataSetX, inputShape, stride, filterShape):
    patchVecs = []
    total = len(dataSetX)
    for i, dataX in enumerate(dataSetX):
        if i % (total // 10) == 0:
            print '%.2f%%' % ((float(i) / float(total)) * 100.)
        patches = get_image_patches(dataX, inputShape, stride, filterShape)

        # Flatten each of the vectors.
        addPatchVecs = [patch.reshape(patch.shape[0] * patch.shape[1] * patch.shape[2]) for patch in patches]
        patchVecs.extend(addPatchVecs)

    with open('data/tmp.h5', 'wb') as f:
        pickle.dump(patchVecs, f)

    return patchVecs


def save_centroids(centroids, filename):
    print 'Saving to file...'
    with open(filename, 'wb') as f:
        pickle.dump(centroids, f)


def load_centroids(filename):
    print 'Loading cluster data...'
    with open(filename, 'rb') as f:
        centroids = pickle.load(f)
        return centroids

def construct_centroids(batch_size, trainSetX, input_shape, stride, filter_shape, k, convolute):
    print '- Building centroids'

    if convolute:
        clusterVecs = build_patch_vecs(trainSetX, input_shape, stride, filter_shape)
    else:
        # Flatten the input.
        sp = trainSetX.shape

        # Not garunteed to be 3 dimensions as the input will be flattened.
        # This is different than performing the convolution where it has to be 3 dimensional.
        input_shape_prod = 1.0
        for input_shape_dim in input_shape:
            input_shape_prod = input_shape_prod * input_shape_dim
        print 'Reshaping'
        print sp
        print input_shape_prod
        clusterVecs = trainSetX.reshape(sp[0], input_shape_prod)

    # mod_batch_size = (len(clusterVecs) // len(trainSetX)) * batch_size
    centroids = kmeans(clusterVecs, k, batch_size)
    if convolute:
        # Expand the output.
        sp = centroids.shape
        return centroids.reshape(sp[0], input_shape[0], filter_shape[0], filter_shape[1])
    else:
        return centroids


def load_or_create_centroids(forceCreate, filename, batch_size, dataSetX, input_shape, stride, filter_shape, k, convolute=True):
    if not forceCreate:
        try:
            centroids = load_centroids(filename)
        except IOError:
            forceCreate = True

    if forceCreate:
        centroids = construct_centroids(batch_size, dataSetX, input_shape, stride, filter_shape, k, convolute)
        save_centroids(centroids, filename)

    return centroids
