# Uncomment these lines if running on macOS it will speed up graphing.
#import matplotlib
#matplotlib.use('Agg')
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import os

from sklearn.manifold import TSNE
import sklearn.preprocessing as preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity

from helpers.mathhelper import get_anchor_vectors
from helpers.mathhelper import convert_onehot_to_index
from helpers.mathhelper import get_anchor_vectors
from helpers.hyper_params import HyperParamData
from helpers.hyper_param_search import HyperParamSearch
from helpers.printhelper import PrintHelper as ph

from add_trainer import AddTrainer

from model_analyzer import ModelAnalyzer
from helpers.printhelper import print_cm

import uuid

def get_hyperparams():
    """
    Get the default hyper parameters.
    A convenience function more than anything.
    """

    # The selection percentages define the (x_i * 100.)% that should be taken
    # at layer i. For instance with the below numbers 30% of the max variance samples
    # will be selected at each layer of the network.
    #selection = [80000, 4000, 5000, None, None]
    cluster_count = 20000
    selection = [int(3 * cluster_count), int(0.8 * cluster_count), 5000, 5000, 5000, None, None]

    # The cluster count is another highly sensitive parameter.
    # The cluster count defines how many of the samples are passed through the
    # network and are used in the k-means calculations to set the anchor vectors.
    # Note that I was able to obtain over 30% accuracy with 30,000 cluster count.
    # I am using 2,000 below because it is faster for testing.

    should_set = [True] * 6

    return HyperParamData(
        input_shape = (1, 28, 28),
        subsample=(1,1),
        patches_subsample = (1,1),
        filter_size=(5,5),
        batch_size = 5,
        nkerns = (6,12,),
        fc_sizes = (120,),
        n_epochs = 10,
        selection_counts = selection,
        activation_func = 'relu',
        extra_path = '',
        should_set_weights = should_set,
        should_eval = True,
        remaining = 100,
        cluster_count = cluster_count)


def search_test():
    ph.DISP = False
    ratios = []
    max_k0 = 6
    max_ratio = 0.0
    #for k0 in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    #        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
    #    hyperparams = get_hyperparams()
    #    hyperparams.extra_path = 'kmeans'
    #    hyperparams.nkers = (k0, 16)
    #    force_create = [True, True, True, True]
    #    model = ModelAnalyzer(hyperparams, force_create=force_create)
    #    model.create_model()
    #    avg_ratio = model.get_avg_ratio()
    #    print(avg_ratio)
    #    if avg_ratio > max_ratio:
    #        max_k0 = k0
    #        max_ratio = avg_ratio

    #    ratios.append(avg_ratio)

    #with open('data/avg_ratio_c0.h5', 'wb') as f:
    #    pickle.dump(ratios, f)

    #ratios = []
    #max_k1 = 0
    #max_ratio = 0.0
    #for k1 in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    #        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
    #    hyperparams = get_hyperparams()
    #    hyperparams.extra_path = 'kmeans'
    #    hyperparams.nkers = (max_k0, k1)
    #    force_create = [True, True, True, True]
    #    model = ModelAnalyzer(hyperparams, force_create=force_create)
    #    model.create_model()
    #    avg_ratio = model.get_avg_ratio()
    #    if avg_ratio > max_ratio:
    #        max_k1 = k1
    #        max_ratio = avg_ratio

    #    ratios.append(avg_ratio)

    #with open('data/avg_ratio_c1.h5', 'wb') as f:
    #    pickle.dump(ratios, f)

    ratios = []
    max_f0 = 0
    max_ratio = 0.0
    for f0 in range(30, 180, 5):
        hyperparams = get_hyperparams()
        hyperparams.extra_path = 'kmeans'
        hyperparams.nkers = (max_k0, max_k1)
        hyperparams.fc_sizes = (f0, 60)
        force_create = [True, True, True, True]
        model = ModelAnalyzer(hyperparams, force_create=force_create)
        model.create_model()
        avg_ratio = model.get_avg_ratio()
        if avg_ratio > max_ratio:
            max_f0 = f0
            max_ratio = avg_ratio

        ratios.append(avg_ratio)

    with open('data/avg_ratio_f0.h5', 'wb') as f:
        pickle.dump(ratios, f)

    ratios = []
    max_f1 = 0
    max_ratio = 0.0
    for f1 in range(20, 80, 5):
        hyperparams = get_hyperparams()
        hyperparams.extra_path = 'kmeans'
        hyperparams.nkers = (max_k0, max_k1)
        hyperparams.fc_sizes = (max_f0, f1)
        force_create = [True, True, True, True]
        model = ModelAnalyzer(hyperparams, force_create=force_create)
        model.create_model()
        avg_ratio = model.get_avg_ratio()
        if avg_ratio > max_ratio:
            max_f1 = f1
            max_ratio = avg_ratio

        ratios.append(avg_ratio)

    with open('data/avg_ratio_f1.h5', 'wb') as f:
        pickle.dump(ratios, f)

    print(max_k0)
    print(max_k1)
    print(max_f0)
    print(max_f1)


def single_test():
    """
    Build a model using the default hyperparameters
    train the model and test the model.
    """

    hyperparams = get_hyperparams()
    hyperparams.extra_path = 'kmeans'
    force_create = [False, False, True, True, True, True]
    model = ModelAnalyzer(hyperparams, force_create=force_create)
    model.create_model()
    model.check_closest()
    #model.adaptive_train()
    #model.adaptive_test()
    #model.eval_performance()
    #model.test_model()
    #model.train_model()
    #model.test_model()
    #model.prune_neurons()
    #model.adaptive_test()
    model.post_eval()

    #add_trainer = AddTrainer(model)
    #add_trainer.disp_clusters()
    #add_trainer.identify_clusters()

    #all_avs = get_anchor_vectors(model)
    #final_avs = all_avs[-1]
    #similarities = cosine_similarity(final_avs)
    #print_cm(similarities, ['%i' % i for i in range(10)])

    #ph.linebreak()

    #av_matching_samples_xy = model.get_closest_anchor_vecs()
    #for i, matching_samples_xy in enumerate(av_matching_samples_xy):
    #    data_dir = 'data/matching_avs/av%i/' % (i)
    #    if not os.path.exists(data_dir):
    #        os.makedirs(data_dir)
    #    for x,y in matching_samples_xy:
    #        print('----' + str(y))
    #        x = x.reshape(x.shape[1], x.shape[2])
    #        plt.imshow(x, cmap='gray')
    #        plt.savefig(data_dir + str(uuid.uuid4()))
    #        plt.clf()

    #    ph.linebreak()

    #print('AV |', end='')
    #for i in range(len(matching_samples_xy)):
    #    print('%{0}i'.format(4) % i, end='')
    #    print('|', end='')

    #print('')

    #print('Val|', end='')
    #for matching_sample_xy in matching_samples_xy:
    #    print('%{0}i'.format(4) % matching_sample_xy[1], end='')
    #    print('|', end='')

    ph.linebreak()

    #model.perform_tsne()
    #model.disp_output_stats()


if __name__ == "__main__":
    single_test()
    #search_test()


