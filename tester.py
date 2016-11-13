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

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import datasets

import csv
import pickle
import os

from clustering import load_or_create_centroids
from clustering import build_patch_vecs
from model_wrapper import ModelWrapper
from helpers.mathhelper import *
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


def fetch_data(test_size):
    dataset = datasets.fetch_mldata('MNIST Original')
    data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
    data = data[:, np.newaxis, :, :]

    return train_test_split(data / 255.0, dataset.target.astype('int'), test_size=test_size)


def save_raw_output(filename, output):
    print 'Saving raw data'
    with open (filename, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        total = len(output)
        sp = np.array(output).shape
        if len(sp) > 2:
            inner_prod = 1
            for dim in sp[1:]:
                inner_prod *= dim
            output = output.reshape(sp[0], int(inner_prod))
        for i, output_vec in enumerate(output):
            writer.writerow(output_vec)


def get_filepaths(extra_path, using_kmeans):
    centroids_out_loc = 'data/centroids/'
    raw_out_loc = 'data/centroids/'

    raw_out_loc += 'python'
    centroids_out_loc += 'python'

    if using_kmeans:
        raw_out_loc += '_kmeans'
        centroids_out_loc += '_kmeans'
    else:
        raw_out_loc += 'reg'
        centroids_out_loc += 'reg'

    raw_out_loc += extra_path
    centroids_out_loc += extra_path
    raw_out_loc += '/raw/'
    centroids_out_loc += '/cluster/'

    if not os.path.exists(raw_out_loc):
        os.makedirs(raw_out_loc)
    if not os.path.exists(centroids_out_loc):
        os.makedirs(centroids_out_loc)

    return (raw_out_loc, centroids_out_loc)


def create_model(train_percentage, should_set_weights, extra_path = '', use_matlab=False, activation_func='relu'):
    # Break the data up into test and training set.
    # This will be set at 0.3 is test and 0.7 is training.
    (train_data, test_data, train_labels, test_labels) = fetch_data(0.3)

    remaining = int(len(train_data) * train_percentage)

    train_labels = np_utils.to_categorical(train_labels, 10)
    test_labels = np_utils.to_categorical(test_labels, 10)

    use_amount = 5000
    train_data = np.array(train_data[0:use_amount])
    train_labels = np.array(train_labels[0:use_amount])
    test_data = np.array(test_data[0:use_amount])
    test_labels = np.array(test_labels[0:use_amount])

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
    force_create = False

    input_centroids = [None] * 5
    layer_out = [None] * 4

    model = Sequential()

    using_kmeans = all(should_set_weight for should_set_weight in should_set_weights)

    raw_out_loc, centroids_out_loc = get_filepaths(extra_path, using_kmeans)

    if force_create:
        save_raw_output(raw_out_loc + 'c0.csv', train_data)

    if should_set_weights[0]:
        print 'Setting conv layer 0 weights'
        tmp_centroids = load_or_create_centroids(force_create, centroids_out_loc + 'c0.csv', batch_size, train_data, input_shape, subsample, filter_size, 6)
        assert tmp_centroids.shape == (6, 25), 'Shape is %s' % str(tmp_centroids.shape)
        input_centroids[0] = tmp_centroids.reshape(6, 1, 5, 5)

    convout0_f = add_convlayer(model, nkerns[0], subsample, filter_size, input_shape=input_shape, weights=input_centroids[0])

    layer_out[0] = convout0_f([train_data])[0]

    if force_create:
        save_raw_output(raw_out_loc + 'c1.csv', layer_out[0])

    if should_set_weights[1]:
        print 'Setting conv layer 1 weights'
        input_shape = (nkerns[0], 14, 14)
        tmp_centroids = load_or_create_centroids(force_create, centroids_out_loc + 'c1.csv', batch_size, layer_out[0], input_shape, subsample, filter_size, 16)
        assert tmp_centroids.shape == (16, 150), 'Shape is %s' % str(tmp_centroids.shape)
        input_centroids[1] = tmp_centroids.reshape(16, 6, 5, 5)

    convout1_f = add_convlayer(model, nkerns[1], subsample, filter_size, input_shape=input_shape, weights=input_centroids[1])
    layer_out[1] = convout1_f([train_data])[0]

    if force_create:
        save_raw_output(raw_out_loc + 'f0.csv', layer_out[1])

    model.add(Flatten())

    if should_set_weights[2]:
        print 'Setting fc layer 0 weights'
        input_shape = (nkerns[1], 7, 7)
        tmp_centroids = load_or_create_centroids(force_create, centroids_out_loc + 'f0.csv', batch_size, layer_out[1], input_shape, subsample, filter_size, 120, convolute=False)
        assert tmp_centroids.shape == (120, 784), 'Shape is %s' % str(tmp_centroids.shape)
        input_centroids[2] = tmp_centroids.reshape(784, 120)

    fc0_f = add_fclayer(model, 120, weights = input_centroids[2])
    layer_out[2] = fc0_f([train_data])[0]

    if force_create:
        save_raw_output(raw_out_loc + 'f1.csv', layer_out[2])

    if should_set_weights[3]:
        print 'Setting fc layer 1 weights'
        input_shape = (120,)
        tmp_centroids = load_or_create_centroids(force_create, centroids_out_loc + 'f1.csv', batch_size, layer_out[2], input_shape, subsample, filter_size, 84, convolute=False)
        assert tmp_centroids.shape == (84, 120), 'Shape is %s' % str(tmp_centroids.shape)
        input_centroids[3] = tmp_centroids.reshape(120, 84)

    fc1_f = add_fclayer(model, 84, weights=input_centroids[3])
    layer_out[3] = fc1_f([train_data])[0]

    if force_create:
        save_raw_output(raw_out_loc + 'f2.csv', layer_out[3])

    if should_set_weights[4]:
        print 'Setting classifier weights'
        input_shape=(84,)
        tmp_centroids = load_or_create_centroids(force_create, centroids_out_loc + 'f2.csv', batch_size, layer_out[3], input_shape, subsample, filter_size, 10, convolute=False)
        assert tmp_centroids.shape == (10, 84), 'Shape is %s' % str(tmp_centroids.shape)
        input_centroids[4] = tmp_centroids.reshape(84, 10)

    classification_layer = Dense(10)
    model.add(classification_layer)

    if should_set_weights[4]:
        bias = classification_layer.get_weights()[1]
        classification_layer.set_weights([input_centroids[4], bias])

    model.add(Activation('softmax'))

    print 'Compiling model'
    opt = SGD(lr = 0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if len(scaled_train_data) > 0:
        anchor_vec_normalizer = AnchorVecNormalizer(filter_size, nkerns)
        model.fit(scaled_train_data, train_labels, batch_size=batch_size, nb_epoch=20, verbose=1, callbacks=[anchor_vec_normalizer])

    (loss, accuracy) = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=1)
    print ''
    print 'Accuracy %.9f%%' % (accuracy * 100.)

    ret_model = ModelWrapper(accuracy, input_centroids, model)

    if not using_kmeans:
        anchor_vecs = get_anchor_vectors(ret_model)
        names = ['c0', 'c1', 'f0', 'f1', 'f2']
        for i, layer_anchor_vecs in enumerate(anchor_vecs):
            save_raw_output(centroids_out_loc + names[i] + '.csv', layer_anchor_vecs)

    return ret_model


def create_models():
    kmeans_model = create_model(0.4, [True] * 5, extra_path = 'whitened')
    # reg_model = create_model(0.0, [False] * 5)


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


    if not model_analyzer.load('data/centroids/python_kmeanswhitened'):
        print 'Could not load models.'
        return False

    model_analyzer.whiten_data(reg_data_stds)
    kmeans_data_means = model_analyzer.get_data_means()
    kmeans_data_stds = model_analyzer.get_data_stds()
    print kmeans_data_means

    model_analyzer.plot_data(reg_data_stds, 'Reg std', 'g')
    model_analyzer.plot_data(kmeans_data_stds, 'Kmeans std', 'y')
    model_analyzer.plot_data(kmeans_data_means, 'Kmeans mean', 'r')
    model_analyzer.plot_data(reg_data_means, 'Reg mean', 'b')

    model_analyzer.show_table()

# create_accuracies()
analyze_accuracies()
