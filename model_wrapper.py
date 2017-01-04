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

import matplotlib.pyplot as plt


class ModelWrapper(object):
    def __init__(self, hyperparams, force_create):
        self.hyperparams = hyperparams
        self.force_create = force_create


    def create_model(self):
        # Break the data up into test and training set.
        # This will be set at 0.3 is test and 0.7 is training.
        (train_data, test_data, train_labels, test_labels) = self.__fetch_data(0.3, 20000)

        #remaining = int(len(train_data) * train_percentage)

        # Only use a given amount of the training data.
        scaled_train_data = train_data[0:self.hyperparams.remaining]
        train_labels = train_labels[0:self.hyperparams.remaining]

        print 'Running for %.2f%% test size' % (train_percentage * 100.)
        print 'The training data has a length of %i' % (len(train_data))

        input_shape           =self.hyperparams.input_shape
        subsample             =self.hyperparams.subsample
        patches_subsample     =self.hyperparams.patches_subsample
        filter_size           =self.hyperparams.filter_size
        batch_size            =self.hyperparams.batch_size
        nkerns                =self.hyperparams.nkerns
        fc_sizes              =self.hyperparams.fc_sizes
        force_create          =self.hyperparams.force_create
        n_epochs              =self.hyperparams.n_epochs
        min_variances         =self.hyperparams.min_variances
        selection_percentages =self.hyperparams.selection_percentages
        use_filters           =self.hyperparams.use_filters

        kmeans_handler = KMeansHandler(should_set_weights, force_create, batch_size, patches_subsample, filter_size, train_data, DiscriminatoryFilter())
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

            if should_set_weights[i]:
                ph.disp('Setting layer weights.')

            is_last = (i == len(nkerns) - 1)
            f_conv_out = self.__add_convlayer(model, nkerns[i], subsample, filter_size,
                            input_shape = input_shape, weights = centroid_weights,
                            flatten=is_last)

            # Pass inputs through see what output is.
            tmp_data = np.empty(input_shape)
            tmp_out = f_conv_out([[tmp_data]])[0]
            input_shape = tmp_out.shape[1:]
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

            if should_set_weights[offset_index]:
                ph.disp('Setting layer weights')

            if i == len(fc_sizes) - 1:
                self.__add_dense_layer(model, fc_sizes[i], weights = centroid_weights)
            else:
                f_fc_out = self.__add_fclayer(model, fc_sizes[i], weights = centroid_weights)

            input_shape = (fc_sizes[i],)
            ph.linebreak()

        model.add(Activation('softmax'))

        ph.disp('Compiling model')
        opt = SGD(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        ph.disp('Model is compiled')

        if len(scaled_train_data) > 0:
            model.fit(scaled_train_data, train_labels, batch_size=batch_size, nb_epoch=n_epochs, verbose=1)

        (loss,accuracy) = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=1)
        print ''
        ph.disp('Accuracy %.9f%%' % (accuracy * 100.), ph.BOLD)
        ph.disp('-' * 50, ph.OKGREEN)
        print '\n' * 2

        ret_model = ModelWrapper(accuracy, None, model)

        return ret_model


        def get_layer_stats(self):
            layer_stats = []
            for layer in self.model.layers:
                # Flatten the layer weights for numerical analysis.
                layer_weights = layer.get_weights()
                if layer_weights is None or len(layer_weights) != 2:
                    continue
                layer_weights = layer_weights[0]

                layer_weights = np.array(layer_weights)
                layer_weights = layer_weights.flatten()

                avg = np.mean(layer_weights)
                std = np.std(layer_weights)
                var = np.var(layer_weights)
                max_val = np.max(layer_weights)
                min_val = np.min(layer_weights)

                layer_stats.append((avg, std, var, max_val, min_val))

            return layer_stats


    def __add_convlayer(model, nkern, subsample, filter_size, flatten=False, input_shape=None, weights=None, activation_func='relu'):
        if input_shape is not None:
            conv_layer = Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same', subsample=subsample, input_shape=input_shape)
        else:
            conv_layer = Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same', subsample=subsample)

        model.add(conv_layer)

        if not weights is None:
            params = conv_layer.get_weights()
            bias = params[1]

            conv_layer.set_weights([weights, bias])

        max_pooling_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
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


    def __add_dense_layer(model, output_dim, weights = None):
        dense_layer = Dense(output_dim)

        model.add(dense_layer)

        if not weights is None:
            bias = dense_layer.get_weights()[1]
            dense_layer.set_weights([weights, bias])


    def __add_fclayer(model, output_dim, weights=None, activation_func='relu'):
        dense_layer = Dense(output_dim)

        model.add(dense_layer)

        if not weights is None:
            bias = dense_layer.get_weights()[1]
            dense_layer.set_weights([weights, bias])

        fcOutLayer = Activation(activation_func)
        model.add(fcOutLayer)
        fcOut_f = K.function([dense_layer.input], [fcOutLayer.output])
        return fcOut_f


    def __fetch_data(test_size, use_amount=None):
        dataset = datasets.fetch_mldata('MNIST Original')
        data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
        data = data[:, np.newaxis, :, :]

        # Seed the random state in the data split.
        (train_data, test_data, train_labels, test_labels) = train_test_split(data / 255.0, dataset.target.astype('int'), test_size=test_size, random_state=42)

        train_labels = np_utils.to_categorical(train_labels, 10)
        test_labels = np_utils.to_categorical(test_labels, 10)

        if use_amount is not None:
            train_data = np.array(train_data[0:use_amount])
            train_labels = np.array(train_labels[0:use_amount])
            test_data = np.array(test_data[0:use_amount])
            test_labels = np.array(test_labels[0:use_amount])

        return (train_data, test_data, train_labels, test_labels)



