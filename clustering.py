from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise
import sklearn.preprocessing as preprocessing
from sklearn.metrics import silhouette_score

from scipy.cluster.vq import whiten

import pickle
import numpy as np
import warnings
import csv
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial

from helpers.printhelper import PrintHelper as ph
from helpers.mathhelper import plot_samples
from helpers.mathhelper import subtract_mean
#from custom_kmeans.k_means_ import KMeans
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
from spherecluster import VonMisesFisherMixture
from scipy.sparse import issparse


def kmeans(input_data, k, batch_size, metric='sp'):
    """
    The actual method to perform k-means.

    :param k: The number of clusters
    :param batch_size: The batch_size used for MiniBatchKMeans
    :param metric: The distance metric to use.

    :returns: The cluster centers.
    """

    ph.disp('Performing %s kmeans on %i vectors' % (metric, len(input_data)), ph.OKBLUE)

    # Check that there are actually enough samples to perform k-means
    if (k > len(input_data) or batch_size > len(input_data)):
        ph.disp('Too few samples for k-means. ' +
                'There are only %i samples while k is %i and batch size is %i' %
                (len(input_data), k, batch_size), ph.FAIL)
        raise ValueError()

    # For the context of this problem only the spherical k-means methods make sense.
    # However, the von mises fisher mixture method is not converging.
    # Therefore, I recommend always using SphericalKMeans

    if metric == 'km':
        km = KMeans(n_clusters=k, n_init=10, n_jobs = -1)

        # Ignore the excessive warnings that sklearn displays
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km.fit(input_data)
        return km.cluster_centers_, km.labels_

    elif metric == 'sp':
        # Spherical clustering.
        skm = SphericalKMeans(n_clusters=k, n_jobs=-1)
        skm.fit(input_data)
        return skm.cluster_centers_, skm.labels_

    elif metric == 'vmfmh':
        # VonMisesFisherMixtureHard
        # I have not been able to get this method to converge.
        vmf_hard = VonMisesFisherMixture(n_clusters=k, n_jobs=-1,posterior_type='hard')
        vmf_hard.fit(input_data)
        return vmf_hard.cluster_centers_

    elif metric == 'mbk':
        # Set the random seed.
        mbk = MiniBatchKMeans(init='k-means++',
                                n_clusters=k,
                                batch_size=batch_size,
                                max_no_improvement=10,
                                reassignment_ratio=0.01,
                                random_state=42,
                                verbose=False)

        # Ignore warnings that sklearn displays
        # for some reason
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mbk.fit(input_data)
        return mbk.cluster_centers_



def get_image_patches(input_img, input_shape, stride, filter_shape):
    """
    For a given 2 dimensional image input extract the sub patches.
    This is supposed to represent the convolution operation.

    :param input_img: The raw input data should be a 2D array.
    :param input_shape: The shape of the input.
    :param stride: The stride of the convolution filter.
    :filter_shape: The shape to sample with.

    :returns: The extracted patches for the input image.
    """

    # Optimized code to extract image subpatches.
    # We are only going to select the first color channel for now.

    all_depth_patches = []
    for depth_channel in range(len(input_img)):
        img_channel = input_img[0]

        # Won't make a copy if not needed
        img_channel = np.ascontiguousarray(input_img[depth_channel])
        X, Y = img_channel.shape
        x, y = filter_shape

        # Check that the stride is actually valid.
        if (X - x) % stride[0] != 0 or (Y - y) % stride[1] != 0:
            raise ValueError('Invalid stride. Non integer number arises!')

        x_dim = (((X-x) / stride[0]) + 1)
        y_dim = (((Y-y) / stride[1]) + 1)
        x_dim = int(x_dim)
        y_dim = int(y_dim)

        # Number of patches, patch_shape
        shape = (x_dim, y_dim, x, y)

        # The right strides can be thought by:
        # 1) Thinking of `img` as a chunk of memory in C order
        # 2) Asking how many items through that chunk of memory are needed when indices
        #    i,j,k,l are incremented by one
        use_strides = img_channel.itemsize * np.array([Y, stride[0], Y, stride[1]])

        patches = np.lib.stride_tricks.as_strided(img_channel, shape=shape, strides=use_strides)
        all_depth_patches.append(patches)

    #all_depth_patches = np.array(all_depth_patches).flatten()

    contiguous_patches = np.ascontiguousarray(all_depth_patches)
    patches_shape = contiguous_patches.shape

    contiguous_patches = contiguous_patches.reshape(patches_shape[1] * patches_shape[2],
            patches_shape[0], patches_shape[3] * patches_shape[4])

    return contiguous_patches


    # Not optimized but clearer code to get image subpatches.
    ## Get the patch.
    #row_offset = 0
    #col_offset = 0
    #patches = []

    ## Remember the receptive field acts across the entire depth parameter.
    #while row_offset <= input_shape[1] - filter_shape[0]:
    #    while col_offset <= input_shape[2] - filter_shape[1]:
    #        patch = []
    #        for filter_mat in input_img:
    #            patch.append(filter_mat[row_offset:row_offset+filter_shape[0],
    #                col_offset:col_offset+filter_shape[1]])

    #        patch = np.array(patch)
    #        patch = patch.flatten()
    #        patches.append(patch)

    #        col_offset += stride[1]

    #    row_offset += stride[0]
    #    col_offset = 0

    #return patches


def build_patch_vecs(data_set_x, input_shape, stride, filter_shape):
    """
    Extracts the image patches for each image. See get_image_patches for more detail.
    This is really more of a wrapper method to print debug statements and act
    across the entire data set rather than just one image.

    :returns: An array of image patches.
    The array dimensions will be (# samples, filter_shape[0], filter_shape[1])
    """

    patch_vecs = []
    total = len(data_set_x)

    display_percent = total / 10
    ph.disp('----Filter shape is ' + str(filter_shape))
    ph.disp('----Stride is ' + str(stride))

    transform_f = partial(get_image_patches, input_shape=input_shape,
            stride=stride, filter_shape=filter_shape)

    # Use concurrent patch extraction.
    # Much faster than the synchronous equivelent.

    with Pool(processes=cpu_count()) as p:
        patch_vecs = p.map(transform_f, data_set_x)

    patch_vecs = np.array(patch_vecs)
    print('----Patch vecs shape ' + str(patch_vecs.shape))
    # This will be a 3D array
    # (# samples, # patches per sample, # flattened filter size dimension)
    patch_vecs_shape = patch_vecs.shape
    patch_vecs = patch_vecs.reshape(patch_vecs_shape[0] * patch_vecs_shape[1],
            patch_vecs_shape[2] * patch_vecs_shape[3])
    print('----Reshaped patch vecs shape ' + str(patch_vecs.shape))

    # The not parralel version of the code above.
    #for i, data_x in enumerate(data_set_x):
    #    if i % display_percent == 0:
    #        ph.disp('----%.2f%%' % ((float(i) / float(len(data_set_x))) * 100.))

    #    patches = get_image_patches(data_x, input_shape, stride, filter_shape)

    #    if i == 0:
    #        ph.disp('Got %i patches for each vector' % (len(patches)), ph.WARNING)
    #        patch_np = np.array(patches[0])
    #        ph.disp('Patch dimension is %s' % (patch_np.shape))
    #    # Add the patches to the culminative list of patches.
    #    patch_vecs.extend(patches)

    #print(np.array(patch_vecs).shape)

    return patch_vecs


def save_centroids(centroids, filename):
    """
    Helper method to save a set of anchor vectors to a filename in CSV format.
    """
    ph.disp('Saving to file...')
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for centroid in centroids:
            writer.writerow(centroid)


def load_centroids(filename):
    """
    Helper method to load a set of anchor vectors from a filename that has CSV data.
    """
    ph.disp('Attempting to load cluster data...')
    centroids = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for centroid in reader:
            centroids.append(centroid)
    return np.array(centroids)

g_layer_cn = 0

def construct_centroids(raw_save_loc, batch_size, train_set_x, input_shape, stride,
        filter_shape, k, convolute, filter_params):
    """
    The entry point for creating the centroids for input samples for a given layer.
    """

    ph.disp('- Building centroids')

    # Do we need to build the image patches because we are in a convolution layer?
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

    cluster_vecs = np.array(cluster_vecs, dtype='float32')

    if raw_save_loc != '':
        ph.disp('Saving image patches')
        with open(raw_save_loc, 'wb') as f:
            csvwriter = csv.writer(f)
            for cluster_vec in cluster_vecs:
                csvwriter.writerow(cluster_vec)

    #TODO:
    # All of these preprocessing steps are very arbitrary.
    # Find the correct preprocessing steps.
    #cluster_vecs = preprocessing.scale(cluster_vecs)

    #cluster_vecs = [preprocessing.scale(cluster_vec) for cluster_vec in cluster_vecs]

    if filter_params is not None:
        cluster_vecs = filter_params.filter_samples(cluster_vecs)

    ph.disp('Mean centering cluster vecs')

    #with Pool(processes=cpu_count()) as p:
    #    cluster_vecs = p.map(subtract_mean, cluster_vecs)

    cluster_vecs = preprocessing.scale(cluster_vecs)

    ph.disp('Cluster vecs centered')

    cluster_vecs = np.array(cluster_vecs)

    ph.disp('Normalizing')
    cluster_vecs = preprocessing.normalize(cluster_vecs, norm='l2')

    ph.disp('Beginning k - means')
    centroids, labels = kmeans(cluster_vecs, k, batch_size)

    ph.disp('Mean centering anchor vectors')
    #centroid_mean = np.mean(centroids)
    #centroids -= centroid_mean

    #centroids = [preprocessing.scale(centroid) for centroid in centroids]
    centroids = preprocessing.scale(centroids)

    centroids = [centroid - np.mean(centroid) for centroid in centroids]
    centroids = np.array(centroids)

    centroids = np.array(centroids)
    centroids = preprocessing.normalize(centroids, norm='l2')

    sample_size = 5000

    cluster_vecs = np.array(cluster_vecs)
    labels = np.array(labels)

    cluster_score = silhouette_score(cluster_vecs, labels, metric = 'cosine', sample_size=sample_size)
    ph.disp('The clustering score is %.3f' % cluster_score, ph.WARNING)
    global g_layer_cn


    if g_layer_cn == 3:
        plot_samples(cluster_vecs, centroids, labels)
        raise ValueError()


    g_layer_cn += 1

    return centroids


def load_or_create_centroids(force_create, filename, batch_size, data_set_x,
        input_shape, stride, filter_shape, k, filter_params, convolute=True,
        raw_save_loc=''):
    """
    Wrapper function to load they anchor vectors for the current layer if they exist
    or otherwise create the anchor vectors. The created centroids will be by default saved.

    :param force_create: Create the anchor vectors even if they already exist at a file location.

    :returns: The calculated or loaded anchor vectors.
    """

    if not force_create:
        try:
            centroids = load_centroids(filename)
            ph.disp('Load succeded')
        except IOError:
            ph.disp('Load failed')
            force_create = True

    if force_create:
        centroids = construct_centroids(raw_save_loc, batch_size, data_set_x, input_shape,
                stride, filter_shape, k, convolute, filter_params)
        save_centroids(centroids, filename)

    return centroids
