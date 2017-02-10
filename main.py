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

from helpers.mathhelper import get_anchor_vectors
from helpers.mathhelper import convert_onehot_to_index
from helpers.mathhelper import get_anchor_vectors
from helpers.hyper_params import HyperParamData
from helpers.hyper_param_search import HyperParamSearch
from helpers.printhelper import PrintHelper as ph

from model_wrapper import ModelWrapper


def get_hyperparams():
    """
    Get the default hyper parameters.
    A convenience function more than anything.
    """

    selection = [0.3, 0.3, 0.3, 0.3, 0.3]
    return HyperParamData(
        input_shape = (1, 28, 28),
        subsample=(1,1),
        patches_subsample = (5,5),
        filter_size=(5,5),
        batch_size = 5,
        nkerns = (6,16),
        fc_sizes = (120, 84, 10,),
        n_epochs = 10,
        min_variances = [0.5, 0.8, 0.8, 0.3, 0.04],
        selection_percentages = selection,
        use_filters = (True, True, True, True, True),
        activation_func = 'relu',
        extra_path = '',
        should_set_weights = [True] * 5,
        should_eval = True,
        remaining = 200,
        cluster_count = 2000)


def single_test():
    """
    Build a model using the default hyperparameters
    train the model and test the model.
    """

    hyperparams = get_hyperparams()
    hyperparams.extra_path = 'kmeans'
    model = ModelWrapper(hyperparams, force_create=False)
    model.create_model()
    model.eval_performance()
    model.train_model()
    model.test_model()
    model.post_eval()


if __name__ == "__main__":
    single_test()

