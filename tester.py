from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD
from keras.utils import np_utils

from helpers.printhelper import PrintHelper

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import datasets

import csv
import pickle
import os

from clustering import build_patch_vecs
from model_wrapper import ModelWrapper
from helpers.mathhelper import *
from kmeans_handler import KMeansHandler
from load_runner import LoadRunner
from anchor_normalizer import AnchorVecNormalizer
from model_analyzer import ModelAnalyzer

import plotly.plotly as py
from plotly.tools import FigureFactory as FF
import plotly.tools as tls

def add_convlayer(model, nkern, subsample, filter_size, input_shape=None, weights=None, activation_func='relu'):
    if input_shape is not None:
        convLayer = Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same', subsample=subsample, input_shape=input_shape)
    else:
        convLayer = Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same', subsample=subsample)

    model.add(convLayer)

    if not weights is None:
        params = convLayer.get_weights()
        bias = params[1]

        convLayer.set_weights([weights, bias])

    model.add(Activation(activation_func))
    max_pooling_out = MaxPooling2D(pool_size=(2,2), strides=(2,2))
    model.add(max_pooling_out)
    convout_f = K.function([model.layers[0].input], [max_pooling_out.output])
    return convout_f


def add_fclayer(model, output_dim, weights=None, activation_func='relu'):
    dense_layer = Dense(output_dim)

    model.add(dense_layer)

    if not weights is None:
        bias = dense_layer.get_weights()[1]
        dense_layer.set_weights([weights, bias])

    fcOutLayer = Activation(activation_func)
    model.add(fcOutLayer)
    fcOut_f = K.function([model.layers[0].input], [fcOutLayer.output])
    return fcOut_f


def fetch_data(test_size, use_amount):
    dataset = datasets.fetch_mldata('MNIST Original')
    data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
    data = data[:, np.newaxis, :, :]

    (train_data, test_data, train_labels, test_labels) = train_test_split(data / 255.0, dataset.target.astype('int'), test_size=test_size)

    train_labels = np_utils.to_categorical(train_labels, 10)
    test_labels = np_utils.to_categorical(test_labels, 10)

    if use_amount is not None:
        train_data = np.array(train_data[0:use_amount])
        train_labels = np.array(train_labels[0:use_amount])
        test_data = np.array(test_data[0:use_amount])
        test_labels = np.array(test_labels[0:use_amount])

    return (train_data, test_data, train_labels, test_labels)

class FilterParams(object):
    CUTOFF = 2000

    def __init__(self, min_variance, selection_percent):
        self.min_variance = min_variance
        self.selection_percent = selection_percent

    def filter_samples(self, samples):
        sample_variances = [(sample, np.var(sample)) for sample in samples]
        variances = [sample_variance[1] for sample_variance in sample_variances]
        sample_variances = [(sample, variance) for sample, variance in sample_variances if variance > self.min_variance]
        selection_count = int(len(sample_variances) * self.selection_percent)
        # order by variance.
        sample_variances = sorted(sample_variances, key = lambda x: x[1])
        samples = [sample_variance[0] for sample_variance in sample_variances]
        samples = samples[0:selection_count]
        if selection_count > self.CUTOFF:
            print '-----Greater than the cutoff randomly sampling'
            selected_samples = []
            for i in np.arange(self.CUTOFF):
                select_index = np.random.randint(len(samples))
                selected_samples.append(samples[select_index])
                del samples[select_index]
            return selected_samples
        else:
            return samples


def create_model(train_percentage, should_set_weights, extra_path = '', activation_func='relu', filter_params = None):
    # Break the data up into test and training set.
    # This will be set at 0.3 is test and 0.7 is training.
    (train_data, test_data, train_labels, test_labels) = fetch_data(0.3, None)

    remaining = int(len(train_data) * train_percentage)

    # Only use a given amount of the training data.
    scaled_train_data = train_data[0:remaining]
    train_labels = train_labels[0:remaining]

    print 'Running for %.2f%% test size' % (train_percentage * 100.)
    print 'The training data has a length of %i' % (len(train_data))

    input_shape = (1, 28, 28)
    subsample=(1,1)
    filter_size=(5,5)
    batch_size = 5
    nkerns = (6, 16)
    fc_sizes = (120, 84, 10)
    force_create = False

    kmeans_handler = KMeansHandler(should_set_weights, force_create, batch_size, subsample, filter_size, train_data, filter_params)
    kmeans_handler.set_filepaths(extra_path)

    model = Sequential()

    f_conv_out = None

    # Create the convolution layers.
    for i in range(len(nkerns)):
        output_shape = (nkerns[i], input_shape[0], filter_size[0], filter_size[1])
        assert_shape = (nkerns[i], input_shape[0] * filter_size[0] * filter_size[1])
        centroid_weights = kmeans_handler.handle_kmeans(i, 'c' + str(i), nkerns[i], input_shape, output_shape,
                                f_conv_out, True, assert_shape = assert_shape)
        f_conv_out = add_convlayer(model, nkerns[i], subsample, filter_size, input_shape = input_shape, weights = centroid_weights)
        input_shape = (nkerns[i], input_shape[1] / 2, input_shape[2] / 2)

    model.add(Flatten())

    f_fc_out = None

    # Create the FC layers.
    for i in range(len(fc_sizes)):
        output_shape = (np.array(input_shape).prod(), fc_sizes[i])
        assert_shape = (fc_sizes[i], np.array(input_shape).prod())
        offset_index = i + len(nkerns)
        centroid_weights = kmeans_handler.handle_kmeans(offset_index, 'f' + str(i), fc_sizes[i], input_shape, output_shape,
                                f_fc_out, False, assert_shape = assert_shape)
        if i == len(fc_sizes) - 1:
            classification_layer = Dense(fc_sizes[i])
            model.add(classification_layer)
        else:
            f_fc_out = add_fclayer(model, fc_sizes[i], weights = centroid_weights)
        input_shape = (fc_sizes[i],)

    model.add(Activation('softmax'))

    print 'Compiling model'
    opt = SGD(lr = 0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if len(scaled_train_data) > 0:
        model.fit(scaled_train_data, train_labels, batch_size=batch_size, nb_epoch=20, verbose=1)

    (loss, accuracy) = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=1)
    print ''
    print 'Accuracy %.9f%%' % (accuracy * 100.)

    ret_model = ModelWrapper(accuracy, None, model)

    return ret_model


def create_models():
    kmeans_model = create_model(0.0, [True] * 5, extra_path='kmeans', filter_params=FilterParams(0.03, 0.5))
    # reg_model = create_model(0.0, [False] * 5, extra_path='reg')


def test_accuracy():
    kmeans_model = create_model(0.4, [True] * 5, extra_path='whitened_cosine')
    print 'Accuracy obtained was ' + str(kmeans_model.accuracy)


def create_accuracies():
    all_accuracies = []
    for use_data in np.arange(0.0, 0.4, 0.1):
        kmeans_model = create_model(use_data, [True] * 5, extra_path='_kmeans_train')
        kmeans_accuracy = kmeans_model.accuracy
        del kmeans_model

        reg_model = create_model(use_data, [False] * 5, extra_path='_reg_train')
        reg_accuracy = reg_model.accuracy
        del reg_model

        all_accuracies.append((kmeans_accuracy, reg_accuracy))

    with open('accuracy_comparison.dat', 'wb') as f:
        pickle.dump(all_accuracies, f)


def analyze_accuracies():
    with open('accuracy_comparison.dat', 'rb') as f:
        all_accuracies = pickle.load(f)

    kmeans_accuracies, reg_accuracies = zip(*all_accuracies)

    model_analyzer = ModelAnalyzer()
    model_analyzer.plot_data(kmeans_accuracies, 'Kmeans accuracy', 'g')
    model_analyzer.plot_data(reg_accuracies, 'Reg accuracy', 'r')
    model_analyzer.show_table()


def analyze_models():
    model_analyzer = ModelAnalyzer()

    if not model_analyzer.load('data/centroids/pythonreg'):
        print 'Could not load reg models'
        return False

    reg_data_means = model_analyzer.get_data_means()
    reg_data_stds = model_analyzer.get_data_stds()
    print reg_data_means


    if not model_analyzer.load('data/centroids/python_kmeans_TEST_sub_mean_norm'):
        print 'Could not load models.'
        return False

    # model_analyzer.whiten_data(reg_data_stds)
    kmeans_data_means = model_analyzer.get_data_means()
    kmeans_data_stds = model_analyzer.get_data_stds()
    print kmeans_data_means

    model_analyzer.plot_data(reg_data_stds, 'Reg std', 'g')
    model_analyzer.plot_data(kmeans_data_stds, 'Kmeans std', 'y')
    model_analyzer.plot_data(kmeans_data_means, 'Kmeans mean', 'r')
    model_analyzer.plot_data(reg_data_means, 'Reg mean', 'b')

    model_analyzer.show_table()

# create_accuracies()
# analyze_accuracies()
create_models()
# analyze_models()
# test_accuracy()
