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
import sklearn.preprocessing as preprocessing
from scipy.spatial.distance import cosine as cosine_dist

import csv
import pickle
import os
import operator

from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial

from clustering import build_patch_vecs
from helpers.mathhelper import *
from kmeans_handler import KMeansHandler

import matplotlib.pyplot as plt


def get_closest_anchor(xy, anchor_vecs):
    x, y = xy
    # Get the closest anchor vector for this sample.
    min_dist = 100000.0
    select_index = -1
    for i, anchor_vec in enumerate(anchor_vecs):
        dist = cosine_dist(x, anchor_vec)
        if dist < min_dist:
            min_dist = dist
            select_index = i

    assert (select_index != -1), "No final layer vectors"
    return (x, y, select_index)


class ModelWrapper(object):
    """
    Encapsulates all of the behavior of the full network.
    Intended to make analyzing and creating the network to be easy.
    """

    def __init__(self, hyperparams, force_create):
        """
        Constructor

        :param force_create: If true nothing will be loaded from memory.
        Everything will be created even if it already exists.
        """
        self.hyperparams        = hyperparams
        self.force_create       = force_create
        self.model              = None
        self.accuracy           = None



    def full_create(self, should_eval=True):
        """
        Shortcut for creating the model remapping the y vals
        and then training and evaluating the model.
        """
        self.create_model()
        if should_eval:
            self.eval_performance()
        self.train_model()
        self.test_model()
        return self.accuracy


    def create_model(self):
        """
        This method is where most of the heavy lifting will happen.
        This method will add the layers to the model and progressively
        compute and set the anchor vector for every single layer.
        """

        # Break the data up into test and training set.
        # This will be set at 0.3 is test and 0.7 is training.
        (train_data, test_data, train_labels, test_labels) = self.__fetch_data(0.3,
                self.hyperparams.cluster_count)

        self.all_train_x = train_data
        self.all_train_y =  train_labels
        self.all_test_x = test_data
        self.all_test_y = test_labels

        # Set all of the hyperparameters to be used.
        input_shape           = self.hyperparams.input_shape
        subsample             = self.hyperparams.subsample
        patches_subsample     = self.hyperparams.patches_subsample
        filter_size           = self.hyperparams.filter_size
        batch_size            = self.hyperparams.batch_size
        nkerns                = self.hyperparams.nkerns
        fc_sizes              = list(self.hyperparams.fc_sizes)
        force_create          = self.force_create
        n_epochs              = self.hyperparams.n_epochs
        selection_percentages = self.hyperparams.selection_percentages
        use_filters           = self.hyperparams.use_filters
        should_set_weights    = self.hyperparams.should_set_weights
        should_eval           = self.hyperparams.should_eval
        extra_path            = self.hyperparams.extra_path

        kmeans_handler = KMeansHandler(should_set_weights, force_create, batch_size,
                patches_subsample, filter_size, train_data, DiscriminatoryFilter())

        kmeans_handler.set_filepaths(extra_path)

        # The Keras model builder.
        model = Sequential()

        f_conv_out = None

        self.__clear_layer_stats()

        # Create the convolution layers.
        for i in range(len(nkerns)):
            if not use_filters[i]:
                kmeans_handler.set_filter_params(None, None)
            else:
                kmeans_handler.set_filter_params(selection_percentages[i])

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
                kmeans_handler.set_filter_params(selection_percentages[offset_index])

            output_shape = (np.array(input_shape).prod(), fc_sizes[i])
            assert_shape = (fc_sizes[i], np.array(input_shape).prod())
            centroid_weights = kmeans_handler.handle_kmeans(offset_index, 'f' + str(i), fc_sizes[i],
                    input_shape, output_shape, f_fc_out, False, assert_shape = assert_shape)

            if centroid_weights.shape[1] != fc_sizes[i]:
                # Made automatic adjustment to the # of clusters.
                ph.disp('Adjusting %i to %i anchor vectors for layer %i' %
                        (fc_sizes[i], centroid_weights.shape[1], offset_index))

                fc_sizes[i] = centroid_weights.shape[1]

            if should_set_weights[offset_index]:
                ph.disp('Setting layer weights')

            if i == len(fc_sizes) - 1:
                self.__add_dense_layer(model, fc_sizes[i], weights = centroid_weights)
            else:
                f_fc_out = self.__add_fclayer(model, fc_sizes[i], weights = centroid_weights)

            input_shape = (fc_sizes[i],)
            ph.linebreak()

        self.final_fc_out = K.function([model.layers[0].input], [model.layers[len(model.layers) - 2].output])

        model.add(Activation('softmax'))

        ph.disp('Compiling model')

        opt = SGD(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        ph.disp('Model is compiled')

        self.model = model


    def train_model(self):
        keras_plot(self.model, to_file='data/model.png', show_shapes=True)
        """
        Train the model the number of samples specified by
        the 'remaining' hyperparameter.
        """
        # Only use a given amount of the training data.
        scaled_train_data = self.all_train_x[0:self.hyperparams.remaining]
        scaled_train_labels = self.all_train_y[0:self.hyperparams.remaining]

        if len(scaled_train_data) > 0:
            self.model.fit(scaled_train_data, scaled_train_labels, batch_size=self.hyperparams.batch_size,
                    nb_epoch=self.hyperparams.n_epochs, verbose=ph.DISP)

        # If you want to be sure the bias is not having a part in here.
        # (it doesn't look like it is)
        #unset_bias(self.model)


    def test_model(self):
        """
        Test and print the accuracy of the model.
        """
        batch_size            = self.hyperparams.batch_size
        (loss, accuracy) = self.model.evaluate(self.all_test_x, self.all_test_y,
                batch_size=batch_size, verbose=ph.DISP)

        ph.linebreak()
        ph.disp('Accuracy %.9f%%' % (accuracy * 100.), ph.BOLD)
        ph.disp('-' * 50, ph.OKGREEN)
        ph.linebreak()

        self.accuracy = accuracy


    def __add_convlayer(self, model, nkern, subsample, filter_size, flatten=False, input_shape=None,
            weights=None, activation_func='relu'):
        """
        Helper method to add a convolution layer.
        """
        if input_shape is not None:
            conv_layer = Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same',
                    subsample=subsample, input_shape=input_shape)
        else:
            conv_layer = Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same',
                    subsample=subsample)

        model.add(conv_layer)

        if not weights is None:
            bias = conv_layer.get_weights()[1]

            conv_layer.set_weights([weights, bias])


        activation_layer = Activation(activation_func)

        model.add(activation_layer)

        max_pooling_out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        model.add(max_pooling_out)

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
        """
        Helper function to add a Dense layer
        """
        print('Output dim ' + str(output_dim))
        dense_layer = Dense(output_dim)

        model.add(dense_layer)

        if not weights is None:
            bias = dense_layer.get_weights()[1]
            print('Weights shape ' + str(weights.shape))
            dense_layer.set_weights([weights, bias])


    def __add_fclayer(self, model, output_dim, weights=None, activation_func='relu'):
        """
        Helper method to add on a FC layer to the model.

        :returns: The function that passes an input through the FC layer.
        """
        dense_layer = Dense(output_dim)

        model.add(dense_layer)

        if not weights is None:
            bias = dense_layer.get_weights()[1]
            dense_layer.set_weights([weights, bias])

        fcOutLayer = Activation(activation_func)
        model.add(fcOutLayer)
        fcOut_f = K.function([dense_layer.input], [fcOutLayer.output])
        return fcOut_f


    def eval_performance(self):
        """
        For unsupervised learning the model will not know which cluster corresponds to
        which number. This is fine in practice but for testing purposes it would be nice
        to use the same evaluation metric to determine the accuracy. Therefore, each of the
        predicted classes will be mapped to an actual class. For instance the after unsupervised
        learning the model might assign the image of a number 2 the label 6. This would be classified
        as a mislabel but in reality the unsupervised system has no knowledge of the underlying
        properties of the labels. This will remap the labels accordingly to give the best accuracy.
        For now this done with a majority voting system.

        :returns: Nothing
        """

        all_train_data = zip(self.all_train_x, self.all_train_y)
        all_pred_y = self.model.predict(self.all_train_x)

        one_hot_pred = []
        # Extract the maximum prediction
        for i in range(len(all_pred_y)):
            pred_y = all_pred_y[i]
            max_index = 0
            for j in range(len(pred_y)):
                if pred_y[j] > pred_y[max_index]:
                    max_index = j
            one_hot_pred.append(max_index)

        one_hot_train = convert_onehot_to_index(self.all_train_y)

        pred_counts = []
        actual_counts = []
        max_val = 9
        min_val = 0
        pred_clusters = []
        for i in range(min_val, max_val + 1):
            comb = zip(one_hot_pred, one_hot_train)
            pred_cluster = [actual for pred, actual in comb if pred == i]
            pred_clusters.append(pred_cluster)

            pred_val_count = len(pred_cluster)
            pred_counts.append(pred_val_count)

            actual_val_count = len([train for train in one_hot_train if train == i])
            actual_counts.append(actual_val_count)

        all_cluster_freq = []
        for cluster_data in pred_clusters:
            cluster_freq = {}
            for actual in cluster_data:
                if actual not in cluster_freq:
                    cluster_freq[actual] = 1
                else:
                    cluster_freq[actual] += 1

            cluster_freq = sorted(cluster_freq.items(), key=operator.itemgetter(1), reverse=True)
            all_cluster_freq.append(cluster_freq)

        accum_freq = []
        for pred, cluster in enumerate(all_cluster_freq):
            accum_freq.extend([(pred, cluster_ele[0], cluster_ele[1]) for cluster_ele in cluster])

        accum_freq = sorted(accum_freq, key = lambda tup: tup[2], reverse=True)

        self.pred_to_actual = {}
        self.actual_to_pred = {}
        while len(self.pred_to_actual) < 10 and len(accum_freq) > 0:
            top = accum_freq.pop(0)
            top_pred = top [0]
            top_actual = top[1]
            top_freq = top[2]

            if top_actual not in self.actual_to_pred and top_pred not in self.pred_to_actual:
                self.actual_to_pred[top_actual] = top_pred
                self.pred_to_actual[top_pred] = top_actual

        if len(self.actual_to_pred) != 10:
            not_existing_pred_entries = list(range(10))
            for actual in self.actual_to_pred:
                pred = self.actual_to_pred[actual]
                if pred in not_existing_pred_entries:
                    not_existing_pred_entries.remove(pred)

            if len(self.actual_to_pred) + len(not_existing_pred_entries) != 10:
                raise ValueError('I programmed this wrong. Sorry.')

            for i in range(10):
                if i not in self.actual_to_pred:
                    not_existing_entry = not_existing_pred_entries.pop()
                    self.actual_to_pred[i] = not_existing_entry
                    self.pred_to_actual[not_existing_entry] = i

        pred_counts = np.array(pred_counts)
        actual_counts = np.array(actual_counts)

        self.pred_dist = pred_counts
        self.actual_dist = actual_counts

        if any(self.hyperparams.should_set_weights):
            ph.disp('Remapping y values', ph.FAIL)
            self.all_train_y = self.__remap_y(self.all_train_y)
            self.all_test_y = self.__remap_y(self.all_test_y)


    def __remap_y(self, y_vals):
        """
        Remap the train_y and test_y values.
        """
        indc_y_vals = convert_onehot_to_index(y_vals)
        mapped_inc_y_vals = [self.actual_to_pred[y_val] for y_val in indc_y_vals]

        return np.array(list(convert_index_to_onehot(mapped_inc_y_vals, 10)))


    def get_closest_anchor_vecs_for_samples(self):
        ph.disp('Getting closest anchor vector for each sample.')
        indicies_y = convert_onehot_to_index(self.all_train_y)

        norm_all_train_x = [np.array(train_x).flatten() for train_x in
                self.all_train_x]

        norm_all_train_x = np.array(norm_all_train_x)

        norm_all_train_x = preprocessing.normalize(norm_all_train_x, norm='l2')

        train_shape = norm_all_train_x.shape
        norm_all_train_x = norm_all_train_x.reshape(train_shape[0], 1, 28, 28)

        # Pass each of the vectors through the network.
        transformed_x = self.final_fc_out([norm_all_train_x])[0]

        # Normalize to the unit sphere.
        transformed_x = preprocessing.normalize(transformed_x, norm='l2')
        self.compare_x = transformed_x

        # Combine into a list containing
        # (image passed through network, numeric value of image)
        train_xy = list(zip(transformed_x, indicies_y))

        # Get the anchor vectors of the network.
        anchor_vecs = get_anchor_vectors(self)

        # Get the anchor vectors of the final layer.
        final_fc_anchor_vecs = anchor_vecs[-1]

        final_fc_anchor_vecs = preprocessing.normalize(final_fc_anchor_vecs,
                norm='l2')

        get_closest_f = partial(get_closest_anchor,
                anchor_vecs = final_fc_anchor_vecs)

        with Pool(processes=cpu_count()) as p:
            sample_anchor_vecs = p.map(get_closest_f, train_xy)

        return sample_anchor_vecs


    def get_closest_anchor_vecs(self, k):
        """
        Get the closest sample to each anchor vector.
        Return the closest sample and the number it corresponds to.

        :returns: A generator producing the sample closest to anchor vector i
        and the value the sample corresponds to.
        """

        ph.disp('Getting closest %i samples to each anchor vector' % k)

        # Convert the one hot vectors to the actual numeric value.
        indicies_y = convert_onehot_to_index(self.all_train_y)

        self.save_indices = indicies_y

        norm_all_train_x = [np.array(train_x).flatten() for train_x in
                self.all_train_x]

        norm_all_train_x = np.array(norm_all_train_x)

        norm_all_train_x = preprocessing.normalize(norm_all_train_x, norm='l2')

        train_shape = norm_all_train_x.shape
        norm_all_train_x = norm_all_train_x.reshape(train_shape[0], 1, 28, 28)

        # Pass each of the vectors through the network.
        transformed_x = self.final_fc_out([norm_all_train_x])[0]

        # Normalize to the unit sphere.
        transformed_x = preprocessing.normalize(transformed_x, norm='l2')
        self.compare_x = transformed_x

        # Combine into a list containing
        # (image passed through network, numeric value of image)
        test_xy = list(zip(transformed_x, indicies_y))

        # Get the anchor vectors of the network.
        anchor_vecs = get_anchor_vectors(self)

        # Get the anchor vectors of the final layer.
        final_fc_anchor_vecs = anchor_vecs[-1]

        final_fc_anchor_vecs = preprocessing.normalize(final_fc_anchor_vecs,
                norm='l2')

        self.final_avs = final_fc_anchor_vecs

        def get_closest_vec(search_vec, test_xy, k = 10):
            min_dists = [1000000.0] * k
            min_indices = [-1] * k

            for i, (test_x, test_y) in enumerate(test_xy):
                dist = cosine_dist(test_x, search_vec)

                for j, min_dist in enumerate(min_dists):
                    if dist < min_dist:
                        min_dists[j] = dist
                        min_indices[j] = i
                        break

            #if min_index == -1:
            #    raise ValueError(('No points in test_xy. There are %i points '
            #                    ' in test_xy') % (len(list(test_xy))))

            return min_indices

        for final_fc_anchor_vec in final_fc_anchor_vecs:
            indices = get_closest_vec(final_fc_anchor_vec, test_xy, k = k)
            yield [(self.all_train_x[i], indicies_y[i]) for i in indices if i != -1]


    def __fetch_data(self, test_size, use_amount=None):
        """
        Get the data and select the correct amount of it.
        """
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


    def __clear_layer_stats(self):
        self.layer_weight_stds = []
        self.layer_weight_avgs = []
        self.layer_anchor_mags_std = []
        self.layer_anchor_mags_avg = []
        self.anchor_vec_spreads_std = []
        self.anchor_vec_spreads_avg = []

        self.layer_bias_stds = []
        self.layer_bias_avgs = []


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





