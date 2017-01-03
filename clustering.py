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
from helpers.printhelper import PrintHelper as ph


def kmeans(input_data, k, batch_size, metric='euclidean'):
    ph.disp('Performing kmeans on %i vectors' % len(input_data), ph.OKBLUE)

    if (k > len(input_data) or batch_size > len(input_data)):
        ph.disp('Too few samples for k-means. ' +
                'There are only %i samples while k is %i and batch size is %i' %
                (len(input_data), k, batch_size), ph.FAIL)
        raise ValueError()

    if (metric == 'cosine'):
        print 'Normalizing'
        input_data = preprocessing.normalize(input_data, norm='l2')
        return cosine_kmeans(input_data, k)
    elif metric == 'euclidean':
        # Set the random seed.
        mbk = MiniBatchKMeans(init='k-means++',
                                random_state=42,
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


def get_image_patches(input_img, input_shape, stride, filter_shape):
    # Reconstruct the image as a matrix.
    # imageMat = inputImg.reshape(inputShape[0], inputShape[1], inputShape[2])

    # Get the patch.
    row_offset = 0
    col_offset = 0
    patches = []

    # Remember the receptive field acts across the entire depth parameter.
    while row_offset <= input_shape[1] - filter_shape[0]:
        while col_offset <= input_shape[2] - filter_shape[1]:
            patch = []
            for filter_mat in input_img:
                patch.append(filter_mat[row_offset:row_offset+filter_shape[0], col_offset:col_offset+filter_shape[1]])

            patch = np.array(patch)
            patch = patch.flatten()
            patches.append(patch)

            col_offset += stride[1]

        row_offset += stride[0]
        col_offset = 0

    return patches


def build_patch_vecs(data_set_x, input_shape, stride, filter_shape):
    patch_vecs = []
    total = len(data_set_x)
    display_percent = total / 10
    for i, data_x in enumerate(data_set_x):
        if i % display_percent == 0:
            print '----%.2f%%' % ((float(i) / float(len(data_set_x))) * 100.)

        patches = get_image_patches(data_x, input_shape, stride, filter_shape)

        if i == 0:
            ph.disp('Got %i patches for each vector' % (len(patches)), ph.WARNING)
        patch_vecs.extend(patches)

    return patch_vecs


def save_centroids(centroids, filename):
    print 'Saving to file...'
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for centroid in centroids:
            writer.writerow(centroid)


def load_centroids(filename):
    print 'Attempting to load cluster data...'
    centroids = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for centroid in reader:
            centroids.append(centroid)
    return np.array(centroids)


def construct_centroids(raw_save_loc, batch_size, train_set_x, input_shape, stride, filter_shape, k, convolute, filter_params):
    print '- Building centroids'
    if convolute:
        ph.disp('--Building patch vecs from %i vectors' % len(train_set_x))
        cluster_vecs = build_patch_vecs(train_set_x, input_shape, stride, filter_shape)
    else:
        # Flatten the input.
        train_set_x = np.array(train_set_x)
        sp = train_set_x.shape

        # Not garunteed to be 3 dimensions as the input will be flattened.
        # This is different than performing the convolution where it has to be 3 dimensional.

        input_shape_prod = 1.0
        for input_shape_dim in input_shape:
            input_shape_prod = input_shape_prod * input_shape_dim
        cluster_vecs = train_set_x.reshape(sp[0], int(input_shape_prod))

    cluster_vecs = np.array(cluster_vecs)

    if raw_save_loc != '':
        print 'Saving image patches'
        with open(raw_save_loc, 'wb') as f:
            csvwriter = csv.writer(f)
            for cluster_vec in cluster_vecs:
                csvwriter.writerow(cluster_vec)

    # print 'Whitening data.'
    # cluster_vecs = whiten(cluster_vecs)

    cluster_vec_mean = np.mean(cluster_vecs)
    cluster_vecs = cluster_vecs - cluster_vec_mean

    if filter_params is not None:
        cluster_vecs = filter_params.filter_samples(cluster_vecs)

    ph.disp('Beginning k - means')
    centroids = kmeans(cluster_vecs, k, batch_size)

    # Normalize each of the centroids.
    for i, centroid in enumerate(centroids):
        centroids[i] = (centroid / np.linalg.norm(centroid))

    ph.disp('Mean centering')
    centroid_mean = np.mean(centroids)
    centroids -= centroid_mean

    return centroids


def load_or_create_centroids(forceCreate, filename, batch_size, dataSetX, input_shape, stride, filter_shape, k, filter_params, convolute=True, scale_factor = 1.0, raw_save_loc=''):
    if not forceCreate:
        try:
            centroids = load_centroids(filename)
            print 'Load succeded'
        except IOError:
            print 'Load failed'
            forceCreate = True

    if forceCreate:
        centroids = construct_centroids(raw_save_loc, batch_size, dataSetX, input_shape, stride, filter_shape, k, convolute, filter_params)
        save_centroids(centroids, filename)

    return centroids
