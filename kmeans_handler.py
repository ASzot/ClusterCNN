import os
import csv
import numpy as np

from helpers.printhelper import PrintHelper as ph

from clustering import load_or_create_centroids


class KMeansHandler(object):
    SHOULD_SAVE_RAW = False

    def __init__(self, should_set_weights, force_create, batch_size,
                    subsample, filter_size, train_data, filter_params):
        self.should_set_weights = should_set_weights
        self.force_create = force_create
        self.batch_size = batch_size
        self.subsample = subsample
        self.filter_size = filter_size
        self.centroids_out_loc = ''
        self.raw_out_loc = ''
        self.train_data = train_data
        self.filter_params = filter_params
        self.prev_out = None


    def set_filter_params(self, min_variance, selection_percent):
        self.filter_params.min_variance = min_variance
        self.filter_params.selection_percent = selection_percent


    def set_filepaths(self, extra_path):
        centroids_out_loc = 'data/centroids/'
        raw_out_loc = 'data/centroids/'

        raw_out_loc += 'python_'
        centroids_out_loc += 'python_'

        raw_out_loc += extra_path
        centroids_out_loc += extra_path
        raw_out_loc += '/raw/'
        centroids_out_loc += '/cluster/'

        if not os.path.exists(raw_out_loc):
            os.makedirs(raw_out_loc)
        if not os.path.exists(centroids_out_loc):
            os.makedirs(centroids_out_loc)

        self.raw_out_loc = raw_out_loc
        self.centroids_out_loc = centroids_out_loc


    def handle_kmeans(self, layer_index, save_name, k, input_shape, output_shape, f_prev_out,
                        convolute, assert_shape = None):
        print_str = ('-' * 10) + ('LAYER %i' % (layer_index)) + ('-' * 10)
        ph.disp(print_str, ph.OKGREEN)

        print 'Input shape ' + str(input_shape)
        print 'Assert shape' + str(assert_shape)
        print 'Output shape ' + str(output_shape)

        if f_prev_out is None:
            # This is the first layer.
            print 'Starting with the training data.'
            layer_out = self.train_data
        else:
            # Chain the output from the previous.
            ph.disp('Chaining from previous output.')
            layer_out = f_prev_out([self.prev_out])[0]

        self.prev_out = layer_out

        if self.SHOULD_SAVE_RAW and self.force_create:
            self.__save_raw_output(self.raw_out_loc + save_name + '.csv', layer_out)

        if self.should_set_weights[layer_index]:
            tmp_centroids = load_or_create_centroids(self.force_create, self.centroids_out_loc +
                save_name + '.csv', self.batch_size, layer_out, input_shape, self.subsample,
                self.filter_size, k, self.filter_params, convolute=convolute)
            if assert_shape is not None:
                assert tmp_centroids.shape == assert_shape, 'Shape is %s' % str(tmp_centroids.shape)
            return tmp_centroids.reshape(output_shape)
        else:
            return None


    def __save_raw_output(self, filename, output):
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
