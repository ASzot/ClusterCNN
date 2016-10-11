from sklearn.cluster import MiniBatchKMeans
import pickle
import numpy as np
import warnings


def kmeans(inputData, k, batch_size):
    mbk = MiniBatchKMeans(init='k-means++',
                        n_clusters=k,
                        batch_size=batch_size,
                        max_no_improvement=10,
                        reassignment_ratio=0.01,
                        verbose=True)

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

def construct_centroids(batch_size, trainSetX, input_shape, stride, filter_shape, k):
    print '- Building centroids'
    patchVecs = build_patch_vecs(trainSetX, input_shape, stride, filter_shape)
    mod_batch_size = (len(patchVecs) // len(trainSetX)) * batch_size
    centroids = kmeans(patchVecs, k, mod_batch_size)
    # Expand each of the vectors.
    sp = centroids.shape
    return centroids.reshape(sp[0], input_shape[0], filter_shape[0], filter_shape[1])


def load_or_create_centroids(forceCreate, filename, batch_size, dataSetX, input_shape, stride, filter_shape, k):
    if not forceCreate:
        try:
            centroids = load_centroids(filename)
        except IOError:
            forceCreate = True

    if forceCreate:
        centroids = construct_centroids(batch_size, dataSetX, input_shape, stride, filter_shape, k)
        save_centroids(centroids, filename)

    return centroids
