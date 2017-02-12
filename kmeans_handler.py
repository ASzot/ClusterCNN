import os
import csv
import numpy as np

from helpers.printhelper import PrintHelper as ph

from clustering import load_or_create_centroids


class KMeansHandler(object):
    # Should save the output of pre k-means to a file?
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


    def set_filter_params(self, selection_percent):
        """
        Set the parameters for the discriminitory filter which
        selects samples based off of variance.
        """
        self.filter_params.selection_percent = selection_percent


    def set_filepaths(self, extra_path):
        """
        Set the filepaths for where the anchor vector information is
        saved to.
        """

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
        """
        Perform k-means for this layer producing the anchor vectors for the layer.

        :param layer_index: This is the overall index of the layer for instance with 2
        conv layers and 3 FC layers for the 1st FC layer the index would be 2.
        :param save_name: The file to save and load the anchor vectors and possibly raw
        output data to.
        :param k: The number of anchor vectors to create. Used in the k-means algorithm.
        :param input_shape: The input dimensions of this layer.
        :param output_shape: The output dimensions of this layer.
        :param f_prev_out: The Theano function that transforms the sample X the output of
        layer layer_index.
        :param convolute: Whether the convolution operator should be applied to the transformed
        samples. Use this for the convolution layers.
        :param assert_shape: Check that the resulting anchor vectors have EXACTLY this shape.
        Note that this could be different than output shape for convolution layers where the input
        has to be expanded.

        :returns: The anchor vectors or None if the anchor vectors are not to be computed.
        """

        ph.linebreak()
        print_str = ('-' * 10) + ('LAYER %i' % (layer_index)) + ('-' * 10)
        ph.disp(print_str, ph.OKGREEN)

        if self.prev_out is None:
            disp_use_count = len(self.train_data)
        else:
            disp_use_count = len(self.prev_out)

        ph.disp('With %i samples avaliable' % disp_use_count)

        ph.disp('Input shape ' + str(input_shape), ph.FAIL)
        ph.disp('Assert shape' + str(assert_shape), ph.FAIL)
        ph.disp('Output shape ' + str(output_shape), ph.FAIL)

        # This is the first layer there is no need to transform any of the data.
        if f_prev_out is None:
            # This is the first layer.
            ph.disp('Starting with the training data.')
            layer_out = self.train_data
        else:
            if self.should_set_weights[layer_index]:
                # Chain the output from the previous.
                ph.disp('Chaining from previous output.')
            # Transform the input to the output of the previous layer.
            layer_out = f_prev_out([self.prev_out])[0]

        # Cache the output as it will need to be chained into the future layers.
        self.prev_out = layer_out

        # Save the transformed input if the flag is set.
        if self.SHOULD_SAVE_RAW and self.force_create:
            self.__save_raw_output(self.raw_out_loc + save_name + '.csv', layer_out)

        # If the anchor vectors should be calculated calculate them.
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
        """
        Helper method to save the transformed input of a layer.
        """

        ph.disp('Saving raw data')
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

