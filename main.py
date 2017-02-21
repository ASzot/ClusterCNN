# Uncomment these lines if running on macOS it will speed up graphing.
#import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pickle
import numpy as np
import random

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

from model_analyzer import ModelAnalyzer
from helpers.printhelper import print_cm

def get_hyperparams():
    """
    Get the default hyper parameters.
    A convenience function more than anything.
    """

    # The selection percentages are an incredibly sensitive
    # hyperparameter.

    #TODO:
    # Make the model less sensitive to the selection percentage

    # The selection percentages define the (x_i * 100.)% that should be taken
    # at layer i. For instance with the below numbers 30% of the max variance samples
    # will be selected at each layer of the network.
    #selection = [0.004, 0.001, 0.01, 0.008, 0.008]
    selection = [0.0008, 0.003, 0.005, 0.003, 0.008]

    # The cluster count is another highly sensitive parameter.
    # The cluster count defines how many of the samples are passed through the
    # network and are used in the k-means calculations to set the anchor vectors.
    # Note that I was able to obtain over 30% accuracy with 30,000 cluster count.
    # I am using 2,000 below because it is faster for testing.

    #TODO:
    # Allow cluster count to scale.

    return HyperParamData(
        input_shape = (1, 28, 28),
        subsample=(1,1),
        patches_subsample = (1,1),
        filter_size=(5,5),
        batch_size = 5,
        nkerns = (6,16),
        fc_sizes = (120, 84, 10,),
        n_epochs = 10,
        selection_percentages = selection,
        use_filters = (True, True, True, True, True),
        activation_func = 'relu',
        extra_path = '',
        should_set_weights = [True] * 5,
        should_eval = True,
        remaining = 0,
        # 22000 is the minimum number needed to run.
        cluster_count = 50000)



def single_test():
    """
    Build a model using the default hyperparameters
    train the model and test the model.
    """

    hyperparams = get_hyperparams()
    hyperparams.extra_path = 'kmeans'
    model = ModelAnalyzer(hyperparams, force_create=True)
    model.create_model()
    model.eval_performance()
    model.train_model()
    model.test_model()
    model.post_eval()

    all_avs = get_anchor_vectors(model)
    final_avs = all_avs[-1]
    similarities = cosine_similarity(final_avs)
    #print_cm(similarities, ['%i' % i for i in range(10)])

    ph.linebreak()

    #av_matching_samples_xy = list(model.get_closest_anchor_vecs())
    #for matching_samples_xy in av_matching_samples_xy:
    #    for x,y in matching_samples_xy:
    #        print('----' + str(y))
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
    model.disp_output_stats()


if __name__ == "__main__":
    single_test()


