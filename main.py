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
    #selection = [0.004, 0.001, 0.01, 0.008, 0.008]
    selection = [25000, 25000, None, None, None]

    # The cluster count is another highly sensitive parameter.
    # The cluster count defines how many of the samples are passed through the
    # network and are used in the k-means calculations to set the anchor vectors.
    # Note that I was able to obtain over 30% accuracy with 30,000 cluster count.
    # I am using 2,000 below because it is faster for testing.

    return HyperParamData(
        input_shape = (1, 28, 28),
        subsample=(1,1),
        patches_subsample = (1,1),
        filter_size=(5,5),
        batch_size = 5,
        nkerns = (6,16),
        fc_sizes = (128, 60,),
        n_epochs = 10,
        selection_counts = selection,
        activation_func = 'relu',
        extra_path = '',
        should_set_weights = [True] * 5,
        should_eval = True,
        remaining = 100,
        cluster_count = 10000)


def single_test():
    """
    Build a model using the default hyperparameters
    train the model and test the model.
    """

    hyperparams = get_hyperparams()
    hyperparams.extra_path = 'kmeans'
    force_create = [True, True, True, True, True]
    model = ModelAnalyzer(hyperparams, force_create=force_create)
    model.create_model()
    model.adaptive_train()
    #model.eval_performance()
    #model.test_model()
    #model.train_model()
    #model.test_model()
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


