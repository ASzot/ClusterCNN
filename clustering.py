from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise
import sklearn.preprocessing as preprocessing
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from theano.tensor.signal import pool
from theano import tensor as T
import theano

from scipy.cluster.vq import whiten

import os
from os import listdir
import pickle
import numpy as np
import warnings
import csv
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial
import collections
import operator
import uuid

from helpers.printhelper import PrintHelper as ph
from helpers.mathhelper import get_closest_vectors
from helpers.mathhelper import get_nearest_neighbors
from helpers.mathhelper import plot_samples
from helpers.mathhelper import subtract_mean
from helpers.mathhelper import get_freq_percents
from helpers.mathhelper import convert_onehot_to_index
#from custom_kmeans.k_means_ import KMeans
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
from spherecluster import VonMisesFisherMixture
from scipy.sparse import issparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def kmeans(input_data, k, batch_size, metric='sp', pre_txt=''):
    """
    The actual method to perform k-means.

    :param k: The number of clusters
    :param batch_size: The batch_size used for MiniBatchKMeans
    :param metric: The distance metric to use.

    :returns: The cluster centers.
    """

    ph.disp(pre_txt + 'Performing %s kmeans on %i vectors %s' % (metric, len(input_data), input_data.shape), ph.OKBLUE)

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

        all_search_data = []
        min_index = -1
        min_cluster_centers = []
        min_labels = []

        cur_index = 0
        for search_k in [k]:
            search_k = int(search_k)
            try:
                skm = SphericalKMeans(n_clusters=search_k, n_jobs=-1)
                skm.fit(input_data)
            except:
                continue
            labels = skm.labels_

            if np.amin(labels) == np.amax(labels):
                cluster_score = 0.0
                ph.disp('All samples belong to cluster ' +
                        str(np.amin(labels)))
            else:
                cluster_score = silhouette_score(input_data, labels, metric = 'cosine',
                        sample_size=5000)


            all_var = []
            for i in range(search_k):
                this_cluster = []
                for j, point in enumerate(input_data):
                    if labels[j] == i:
                        this_cluster.append(point)

                if len(this_cluster) > 0.0:
                    this_cluster_var = np.var(this_cluster)
                    all_var.append(this_cluster_var)
                else:
                    all_var.append(0.0)

            avg_var = np.mean(all_var)

            ph.disp(pre_txt + '|   search k at %i got %.6f, %.4f' % (search_k, avg_var,
                cluster_score))

            if min_index == -1 or cluster_score > all_search_data[min_index][0]:
                min_index = cur_index
                min_cluster_centers = skm.cluster_centers_
                min_labels = labels

            all_search_data.append((cluster_score, avg_var))

            cur_index += 1

        #with open('data/' + str(uuid.uuid4()) + '.dat') as f:
        #    pickle.dump(all_search_data, f)

        return min_cluster_centers, min_labels

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

        labels = mbk.labels_
        cluster_score = silhouette_score(input_data, labels, metric = 'euclidean',
                sample_size=5000)

        ph.disp('Got a euclidean ss of %.2f' % (cluster_score))

        return mbk.cluster_centers_, labels



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
    ph.disp('----Patch vecs shape ' + str(patch_vecs.shape))
    # This will be a 3D array
    # (# samples, # patches per sample, # flattened filter size dimension)
    patch_vecs_shape = patch_vecs.shape
    patch_vecs = patch_vecs.reshape(patch_vecs_shape[2], patch_vecs_shape[0] * patch_vecs_shape[1],
            patch_vecs_shape[3])
    ph.disp('----Reshaped patch vecs shape ' + str(patch_vecs.shape))

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


def plot_silhouette_scores(cluster_score, samples_scores, should_plot=False):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    y_lower = 10
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = samples_scores[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / k)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                0, ith_cluster_silhouette_values,
                facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.axvline(x=cluster_score, color="red", linestyle="--")
    ax1.set_yticks([])

    if should_plot:
        plot_samples(layer_cluster_vecs, layer_centroids, labels, show_plt=ax2)

    plt.show()


def save_raw_image_patches(cluster_vecs, raw_save_loc):
    ph.disp('Saving image patches')
    with open(raw_save_loc, 'wb') as f:
        csvwriter = csv.writer(f)
        for cluster_vec in cluster_vecs:
            csvwriter.writerow(cluster_vec)


def build_cluster_vecs(train_set_x, input_shape, stride, filter_shape,
        convolute):
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
        # Wrap in another dimension.
        cluster_vecs = train_set_x.reshape(1, sp[0], int(input_shape_prod))

    return np.array(cluster_vecs, dtype='float32')


def post_process_centroids(centroids):
    centroids = np.array(centroids)
    #print('')
    #print('CENTROID BEFORE PROC')
    #print('Min ' + str(np.amin(centroids)) + ', ', end='')
    #print('Max ' + str(np.amax(centroids)) + ', ', end='')
    #print('Mean ' + str(np.mean(centroids)) + ', ', end='')
    #print('STD ' +  str(np.std(centroids)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if np.isnan(np.sum(centroids)):
            raise ValueError('Is NaN')
        centroids = centroids.astype(np.float64)
        centroids = preprocessing.scale(centroids)
        #centroids = subtract_mean(centroids)
        centroids = preprocessing.normalize(centroids, norm='l2')

    #print('CENTROID AFTER PROC')
    #print('Min ' + str(np.amin(centroids)) + ', ', end='')
    #print('Max ' + str(np.amax(centroids)) + ', ', end='')
    #print('Mean ' + str(np.mean(centroids)) + ', ', end='')
    #print('STD ' +  str(np.std(centroids)))
    #print('')

    return centroids


def pre_process_clusters(cluster_vecs, convolute):
    orig_shape = None
    if len(cluster_vecs.shape) > 2:
        orig_shape = cluster_vecs.shape
        cluster_vecs = cluster_vecs.reshape(-1, orig_shape[1] * orig_shape[2] * orig_shape[3])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #cluster_vecs = preprocessing.scale(cluster_vecs)
        cluster_vecs = preprocessing.normalize(cluster_vecs, norm='l2')

    if orig_shape is not None:
        cluster_vecs = cluster_vecs.reshape(*orig_shape)

    return cluster_vecs


def post_sort_process_clusters(cluster_vecs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #cluster_vecs = preprocessing.scale(cluster_vecs)
    return cluster_vecs


def recur_apply_kmeans(layer_cluster_vecs, k, batch_size, min_cluster_samples,
        max_std, can_recur, all_train_y, all_train_x, mappings, cur_layer,
        model, right, wrong, branch_depth = 0):

    if branch_depth == 1:
        can_recur = False

    #print('')
    #print('INCOMING CLUSTER DATA')
    #print('Min ' + str(np.amin( layer_cluster_vecs)) + ', ', end='')
    #print('Max ' + str(np.amax( layer_cluster_vecs)) + ', ', end='')
    #print('Mean ' + str(np.mean(layer_cluster_vecs)) + ', ', end='')
    #print('STD ' +  str(np.std( layer_cluster_vecs)))
    #print('')

    #layer_cluster_vecs = whiten(layer_cluster_vecs)

    #if cur_layer == 3:
        #plot_samples(layer_cluster_vecs, None, all_train_y)
        #raise ValueError()

    pre_txt = '---' * branch_depth
    ph.disp('')
    ph.disp(pre_txt + 'At branch depth %i' % branch_depth)
    layer_centroids, labels = kmeans(layer_cluster_vecs, k, batch_size, pre_txt = pre_txt)
    # We will compute our own labels.
    #ph.disp('There are %i centroids %i layer cluster_vecs and %i y train samples'
    #        % (len(layer_centroids), len(layer_cluster_vecs),
    #            len(all_train_y)))

    layer_centroids = post_process_centroids(layer_centroids).tolist()

    if cur_layer != (len(model.hyperparams.nkerns) + len(model.hyperparams.fc_sizes) - 1):
        return layer_centroids
    #if cur_layer < 2:
    #    return layer_centroids

    closest_anchor_vecs = get_closest_vectors(layer_centroids, list(zip(layer_cluster_vecs,
        all_train_y)))

    #labels = [closest_anchor_vec[2] for closest_anchor_vec in
    #        closest_anchor_vecs]

    #labels = np.array(labels)

    overall_label_freqs = list(get_freq_percents(labels))
    overall_label_freqs = sorted(overall_label_freqs, key=lambda x: x[1], reverse=True)

    freq_margin = int(len(layer_cluster_vecs) * 0.75)
    if overall_label_freqs[0][1] > freq_margin:
        can_recur = False

    final_centroids = []
    all_ratios = []

    compute_nn = False

    if compute_nn:
        ph.disp('Computing NN')
        N = 5
        like_freqs = {}
        for j in range(N + 1):
            like_freqs[j] = 0
        nbrs = NearestNeighbors(n_neighbors=N, algorithm='ball_tree').fit(layer_cluster_vecs)

    for i in range(k):
        this_cluster = []
        real_labels = []
        real_samples = []
        for j, label in enumerate(labels):
            if label == i:
                this_cluster.append(layer_cluster_vecs[j])
                real_labels.append(all_train_y[j])
                real_samples.append(all_train_x[j])

        label_freqs = list(get_freq_percents(real_labels))
        label_freqs = sorted(label_freqs, key=lambda x: x[1], reverse=True)
        print(pre_txt + str(len(this_cluster)) + '_' + str(label_freqs))

        disp_str = ''
        if len(label_freqs) > 0:
            disp_str += ('Out of %i: %i correct ' % (len(this_cluster), label_freqs[0][1]))
        else:
            disp_str += 'Cluster %i had no vectors associated with it' % i

        if len(label_freqs) > 1:
            disp_str += ('and %i incorrect' % (sum([af[1] for af in
                label_freqs[1:]])))

        #ph.disp(disp_str)

        this_cluster = np.array(this_cluster)

        if compute_nn:
            distances, nearest_indices_all = nbrs.kneighbors(this_cluster)
            for nearest_indices in nearest_indices_all:
                count_equal = 0
                for nearest_index in nearest_indices:
                    cur_label = labels[nearest_index]
                    if cur_label== i:
                        count_equal += 1
                like_freqs[count_equal] += 1

        this_cluster_std = 0.0
        if len(this_cluster) > 0:
            this_cluster_std = np.var(this_cluster)
            this_cluster_avg = np.mean(this_cluster)

        if (len(label_freqs) > 0 and label_freqs[0][1] / float(len(this_cluster))) < 0.6:
            disp_color = ph.FAIL
        else:
            disp_color = ph.OKGREEN

        # Should divide the cluster even further?
        #if can_recur and len(this_cluster) > min_cluster_samples and max_std < this_cluster_std:
        if can_recur and len(this_cluster) > min_cluster_samples:
            #ph.linebreak()
            ph.disp(pre_txt + 'Branching cluster')

            sub_mapping = {}
            this_cluster = pre_process_clusters(this_cluster, False)
            k = 10
            sub_layer_centroids = recur_apply_kmeans(this_cluster, k,
                    batch_size, min_cluster_samples, max_std, can_recur,
                    real_labels, all_train_x, sub_mapping, cur_layer, model,
                    right, wrong, branch_depth + 1)
            #ph.linebreak()

            mappings[i] = sub_mapping
            final_centroids.extend(sub_layer_centroids)
        else:
            if len(label_freqs) > 0:
                right.append(label_freqs[0][1])
            if len(label_freqs) > 1:
                wrong.extend([af[1] for af in label_freqs[1:]])
            mappings[i] = real_samples
            final_centroids.append(layer_centroids[i])

    if compute_nn:
        ph.disp('Done computing NN')

    cluster_acc = float(sum(right)) / float(sum(right) + sum(wrong))
    ph.disp(pre_txt + 'Avg Ratio %.2f' % (cluster_acc * 100.))
    model.set_avg_ratio(cluster_acc)

    if compute_nn:
        total_like_freqs = sum(like_freqs.values())
        for j in range(N + 1):
            fraction_like = float(like_freqs[j]) / float(total_like_freqs)
            ph.disp('%.2f of samples have %i/%i of same samples closest' %
                    (fraction_like * 100., j, N))

    return final_centroids


def apply_kmeans(layer_cluster_vecs, k, cur_layer, model_wrapper, batch_size):
    layer_cluster_vecs = post_sort_process_clusters(layer_cluster_vecs)
    #layer_cluster_vecs = pre_process_clusters(layer_cluster_vecs)
    #if cur_layer == 2:
    #    plot_samples(layer_cluster_vecs, None, [0] * len(layer_cluster_vecs))

    ph.disp('The cur layer is %i' % cur_layer)
    max_std = 0.01
    min_samples_percentage = 0.01

    # The minimum # of samples per cluster.
    # Note that this rule always has precedence over the max std rule.
    min_cluster_samples = int(len(layer_cluster_vecs) * min_samples_percentage)
    min_cluster_samples = 1000

    can_recur = (cur_layer == 2)
    #can_recur = False
    #can_recur = cur_layer > 1

    if can_recur:
        ph.disp('The max std per cluster:       %.4f' % max_std)
        ph.disp('Min # of samples per cluster:  %i' % min_cluster_samples)

    train_y = convert_onehot_to_index(model_wrapper.all_train_y)

    mapping = {}
    right = []
    wrong = []
    all_centroids = recur_apply_kmeans(layer_cluster_vecs, k, batch_size,
            min_cluster_samples, max_std, can_recur, train_y,
            model_wrapper.all_train_x, mapping, cur_layer, model_wrapper,
            right, wrong)

    model_wrapper.set_mapping(mapping)

    return np.array(all_centroids, np.float64)


def construct_centroids(raw_save_loc, batch_size, train_set_x, input_shape, stride,
        filter_shape, k, convolute, filter_params, layer_index, model_wrapper):
    """
    The entry point for creating the centroids for input samples for a given layer.
    """

    cluster_vecs = build_cluster_vecs(train_set_x, input_shape, stride,
            filter_shape, convolute)

    cvs = cluster_vecs.shape
    cluster_vecs = cluster_vecs.reshape(cvs[1], cvs[0] * cvs[2])

    print('')
    print('PRE PROC CLUSTER DATA')
    print('Min ' + str(np.amin( cluster_vecs)) + ', ', end='')
    print('Max ' + str(np.amax( cluster_vecs)) + ', ', end='')
    print('Mean ' + str(np.mean(cluster_vecs)) + ', ', end='')
    print('STD ' +  str(np.std( cluster_vecs)))
    print('')

    cluster_vecs = pre_process_clusters(cluster_vecs, convolute)

    print('')
    print('POST PROC CLUSTER DATA')
    print('Min ' + str(np.amin( cluster_vecs)) + ', ', end='')
    print('Max ' + str(np.amax( cluster_vecs)) + ', ', end='')
    print('Mean ' + str(np.mean(cluster_vecs)) + ', ', end='')
    print('STD ' +  str(np.std( cluster_vecs)))
    print('')

    if raw_save_loc != '':
        save_raw_image_patches(cluster_vecs, raw_save_loc)

    if convolute:
        cluster_vecs = filter_params.get_sorted(cluster_vecs, layer_index)

        #cluster_vecs = np.array(list(filter_params.filter_outliers(cluster_vecs)))

        #cluster_vecs = cluster_vecs.reshape(cvs[0], -1, cvs[2])
        #cs = cluster_vecs.shape
        #cluster_vecs = cluster_vecs.reshape(-1, cs[0] * cs[2])
    else:
        cluster_vecs = filter_params.get_selected(cluster_vecs, layer_index)
        #cluster_vecs = filter_params.get_sorted(cluster_vecs, layer_index)

    cluster_vecs = np.array(cluster_vecs)

    centroids = apply_kmeans(cluster_vecs, k, layer_index,
            model_wrapper, batch_size)

    ph.disp('Centroids now have shape %s' % str(centroids.shape))

    return centroids


def load_or_create_centroids(force_create, filename, batch_size, data_set_x,
        input_shape, stride, filter_shape, k, filter_params, layer_index,
        model_wrapper, convolute=True, raw_save_loc=''):
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
                stride, filter_shape, k, convolute, filter_params, layer_index,
                model_wrapper)
        save_centroids(centroids, filename)

    return centroids

