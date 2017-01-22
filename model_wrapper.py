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
from helpers.mathhelper import *
from kmeans_handler import KMeansHandler
from load_runner import LoadRunner
from model_analyzer import ModelAnalyzer

import matplotlib.pyplot as plt


class ModelWrapper(object):
    def __init__(self, hyperparams, force_create):
        self.hyperparams        = hyperparams
        self.force_create       = force_create
        self.model              = None
        self.accuracy           = None


    def set_hyperparams(self, hyperparams):
        self.hyperparams = hyperparams


    def set_hyperparam(self, hyperparam_name, hyperparam_value):
        if hyperparam_name.startswith('min_variances_') or hyperparam_name.startswith('selection_percentages_'):
            name_parts = hyperparam_name.split('_')

            if len(name_parts) != 3:
                raise ValueError('Invalid hyper param name')

            if isinstance(name_parts[2], int):
                raise ValueError('Invalid index supplied to hyper param name')

            attr_name = name_parts[0] + '_' + name_parts[1]
            attr_value = getattr(self.hyperparams, attr_name)
            attr_index = int(name_parts[2])

            attr_value[attr_index] = hyperparam_value
            self.set_hyperparam(attr_name, attr_value)
        else:
            setattr(self.hyperparams, hyperparam_name, hyperparam_value)


    def eval_performance(self):
        all_train_data = zip(self.all_train_x, self.all_train_y)
        all_pred_y = self.model.predict(self.all_train_x)

        one_hot_pred = []
        for i in range(len(all_pred_y)):
            pred_y = all_pred_y[i]
            max_index = 0
            for j in range(len(pred_y)):
                if pred_y[j] > pred_y[max_index]:
                    max_index = j
            one_hot_pred.append(max_index)

        one_hot_train = []
        for i in range(len(self.all_train_y)):
            start_count = len(one_hot_train)

            for j, train_y in enumerate(self.all_train_y[i]):
                if train_y != 0:
                    one_hot_train.append(j)

            if len(one_hot_train) == start_count:
                print 'New elements have not been appended'
                return

        pred_counts = []
        actual_counts = []
        max_val = 9
        min_val = 0
        for i in range(min_val, max_val + 1):
            pred_val_count = len([pred for pred in one_hot_pred if pred == i])
            pred_counts.append(pred_val_count)

            actual_val_count = len([train for train in one_hot_train if train == i])
            actual_counts.append(actual_val_count)

        pred_counts = np.array(pred_counts)
        actual_counts = np.array(actual_counts)

        self.pred_dist = pred_counts
        self.actual_dist = actual_counts


    def create_model(self):
        # Break the data up into test and training set.
        # This will be set at 0.3 is test and 0.7 is training.
        (train_data, test_data, train_labels, test_labels) = self.__fetch_data(0.3, 1000)
        self.all_train_x = train_data
        self.all_train_y =  train_labels

        #remaining = int(len(train_data) * train_percentage)

        # Only use a given amount of the training data.
        scaled_train_data = train_data[0:self.hyperparams.remaining]
        train_labels = train_labels[0:self.hyperparams.remaining]

        input_shape           = self.hyperparams.input_shape
        subsample             = self.hyperparams.subsample
        patches_subsample     = self.hyperparams.patches_subsample
        filter_size           = self.hyperparams.filter_size
        batch_size            = self.hyperparams.batch_size
        nkerns                = self.hyperparams.nkerns
        fc_sizes              = self.hyperparams.fc_sizes
        force_create          = self.force_create
        n_epochs              = self.hyperparams.n_epochs
        min_variances         = self.hyperparams.min_variances
        selection_percentages = self.hyperparams.selection_percentages
        use_filters           = self.hyperparams.use_filters
        should_set_weights    = self.hyperparams.should_set_weights
        should_eval           = self.hyperparams.should_eval
        extra_path            = self.hyperparams.extra_path

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
            model.fit(scaled_train_data, train_labels, batch_size=batch_size, nb_epoch=n_epochs, verbose=ph.DISP)

        if should_eval:
            (loss, accuracy) = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=ph.DISP)
        else:
            accuracy = 0.0

        ph.linebreak()
        ph.disp('Accuracy %.9f%%' % (accuracy * 100.), ph.BOLD)
        ph.disp('-' * 50, ph.OKGREEN)
        ph.linebreak()

        self.accuracy = accuracy
        self.model    = model


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


    def __add_convlayer(self, model, nkern, subsample, filter_size, flatten=False, input_shape=None, weights=None, activation_func='relu'):
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


    def __add_dense_layer(self, model, output_dim, weights = None):
        dense_layer = Dense(output_dim)

        model.add(dense_layer)

        if not weights is None:
            bias = dense_layer.get_weights()[1]
            dense_layer.set_weights([weights, bias])


    def __add_fclayer(self, model, output_dim, weights=None, activation_func='relu'):
        dense_layer = Dense(output_dim)

        model.add(dense_layer)

        if not weights is None:
            bias = dense_layer.get_weights()[1]
            dense_layer.set_weights([weights, bias])

        fcOutLayer = Activation(activation_func)
        model.add(fcOutLayer)
        fcOut_f = K.function([dense_layer.input], [fcOutLayer.output])
        return fcOut_f


    def __fetch_data(self, test_size, use_amount=None):
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



