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

from helpers.printhelper import PrintHelper as ph
from model_layers.discriminatory_filter import DiscriminatoryFilter

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
from model_analyzer import ModelAnalyzer

import plotly.plotly as py
from plotly.tools import FigureFactory as FF
import plotly.tools as tls

def add_convlayer(model, nkern, subsample, filter_size, flatten = False, input_shape=None, weights=None, activation_func='relu'):
    if input_shape is not None:
        conv_layer = Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same', subsample=subsample, input_shape=input_shape)
    else:
        conv_layer = Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same', subsample=subsample)

    model.add(conv_layer)

    if not weights is None:
        params = conv_layer.get_weights()
        bias = params[1]

        conv_layer.set_weights([weights, bias])

    max_pooling_out = MaxPooling2D(pool_size=(2,2), strides=(2,2))
    activation_layer = Activation(activation_func)

    model.add(max_pooling_out)
    model.add(activation_layer)

    if flatten:
        ph.disp('Flattening output')
        flatten_layer = Flatten()
        model.add(flatten_layer)
        output = flatten_layer.output
    else:
        output = max_pooling_out.output

    # The function is the output of the conv / pooling layers.
    convout_f = K.function([conv_layer.input], [output])
    return convout_f


def add_fclayer(model, output_dim, weights=None, activation_func='relu'):
    dense_layer = Dense(output_dim)

    model.add(dense_layer)

    if not weights is None:
        bias = dense_layer.get_weights()[1]
        dense_layer.set_weights([weights, bias])

    fcOutLayer = Activation(activation_func)
    model.add(fcOutLayer)
    fcOut_f = K.function([dense_layer.input], [fcOutLayer.output])
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


def create_model(train_percentage, should_set_weights, extra_path = '', activation_func='relu'):
    # Break the data up into test and training set.
    # This will be set at 0.3 is test and 0.7 is training.
    (train_data, test_data, train_labels, test_labels) = fetch_data(0.3, 10000)

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
    force_create = True
    n_epochs = 10
    min_variances = (0.03, 0.3, 4., 50., 0.6)
    selection_percentages = np.arange(0.75, 1.0, 0.05)
    selection_percentages[0] = 0.2
    use_filters = (False, False, False, False, False)

    kmeans_handler = KMeansHandler(should_set_weights, force_create, batch_size, subsample, filter_size, train_data, DiscriminatoryFilter())
    kmeans_handler.set_filepaths(extra_path)

    model = Sequential()

    f_conv_out = None

    # Create the convolution layers.
    for i in range(len(nkerns)):
        if not use_filters[i]:
            kmeans_handler.set_filter_params(None, None)
        else:
            kmeans_handler.set_filter_params(min_variances[i], selection_percentages[i])

        output_shape = (nkerns[i], input_shape[0], filter_size[0], filter_size[1])
        assert_shape = (nkerns[i], input_shape[0] * filter_size[0] * filter_size[1])
        centroid_weights = kmeans_handler.handle_kmeans(i, 'c' + str(i), nkerns[i], input_shape, output_shape,
                                f_conv_out, True, assert_shape = assert_shape)
        ph.disp('Setting layer weights.')
        is_last = (i == len(nkerns) - 1)
        f_conv_out = add_convlayer(model, nkerns[i], subsample, filter_size,
                        input_shape = input_shape, weights = centroid_weights,
                        flatten=is_last)
        input_shape = (nkerns[i], input_shape[1] / 2, input_shape[2] / 2)
        ph.linebreak()

    f_fc_out = f_conv_out

    # Create the FC layers.
    for i in range(len(fc_sizes)):
        offset_index = i + len(nkerns)
        if not use_filters[offset_index]:
            kmeans_handler.set_filter_params(None, None)
        else:
            kmeans_handler.set_filter_params(min_variances[offset_index], selection_percentages[offset_index])

        output_shape = (np.array(input_shape).prod(), fc_sizes[i])
        assert_shape = (fc_sizes[i], np.array(input_shape).prod())
        centroid_weights = kmeans_handler.handle_kmeans(offset_index, 'f' + str(i), fc_sizes[i], input_shape, output_shape,
                                f_fc_out, False, assert_shape = assert_shape)
        if i == len(fc_sizes) - 1:
            classification_layer = Dense(fc_sizes[i])
            model.add(classification_layer)
        else:
            ph.disp('Setting layer weights')
            f_fc_out = add_fclayer(model, fc_sizes[i], weights = centroid_weights)
        input_shape = (fc_sizes[i],)
        ph.linebreak()

    model.add(Activation('softmax'))

    ph.disp('Compiling model')
    opt = SGD(lr = 0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    ph.disp('Model is compiled')

    if len(scaled_train_data) > 0:
        model.fit(scaled_train_data, train_labels, batch_size = batch_size, nb_epoch=n_epochs, verbose=1)

    (loss, accuracy) = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=1)
    print ''
    print 'Accuracy %.9f%%' % (accuracy * 100.)

    ret_model = ModelWrapper(accuracy, None, model)

    return ret_model


def create_models():
    kmeans_model = create_model(0.1, [True] * 5, extra_path='kmeans')
    # reg_model = create_model(0.1, [False] * 5, extra_path='reg')


def test_accuracy():
    kmeans_model = create_model(0.2, [True] * 5, extra_path='whitened_cosine')
    print 'Accuracy obtained was ' + str(kmeans_model.accuracy)


def create_accuracies():
    all_accuracies = []
    for use_data in np.arange(0.0, 0.4, 0.05):
        kmeans_model = create_model(use_data, [True] * 5, extra_path='kmeans_train')
        kmeans_accuracy = kmeans_model.accuracy
        ph.disp('Accuracy for kmeans %.9f%%' % (kmeans_accuracy), ph.HEADER)
        del kmeans_model

        reg_model = create_model(use_data, [False] * 5, extra_path='reg_train')
        reg_accuracy = reg_model.accuracy
        ph.disp('Accuracy for regular %.9f%%' % (reg_accuracy), ph.HEADER)
        del reg_model

        all_accuracies.append((kmeans_accuracy, reg_accuracy))

    with open('data/accuracies/accuracy_comparison.dat', 'wb') as f:
        pickle.dump(all_accuracies, f)


def analyze_accuracies():
    with open('data/accuracies/accuracy_comparison.dat', 'rb') as f:
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
