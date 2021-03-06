import os
import csv
from keras import backend as K
import numpy as np

from helpers.printhelper import PrintHelper as ph
from clustering import pre_process_clusters

from clustering import load_or_create_centroids


class KMeansHandler(object):
    # Should save the output of pre k-means to a file?
    SHOULD_SAVE_RAW = False

    def __init__(self, should_set_weights, force_create, batch_size,
                    subsample, filter_size, train_data, filter_params,
                    model_wrapper):
        self.should_set_weights = should_set_weights
        self.force_create = force_create
        self.batch_size = batch_size
        self.subsample = subsample
        self.filter_size = filter_size
        self.centroids_out_loc = ''
        self.raw_out_loc = ''
        self.train_data = train_data
        self.filter_params = filter_params
        self.model_wrapper = model_wrapper


    def set_filter_params(self, selection_count):
        """
        Set the parameters for the discriminitory filter which
        selects samples based off of variance.
        """
        self.filter_params.selection_count = selection_count


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


    def handle_kmeans(self, layer_index, save_name, k, input_shape, output_shape,
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

        ph.disp('Input shape ' + str(input_shape), ph.FAIL)
        ph.disp('Assert shape' + str(assert_shape), ph.FAIL)
        ph.disp('Output shape ' + str(output_shape), ph.FAIL)

        f_prev_out = None
        wrapper_model = self.model_wrapper.model

        if len(wrapper_model.layers) > 0:
            f_prev_out = K.function([wrapper_model.layers[0].input],
                    [wrapper_model.layers[len(wrapper_model.layers) - 1].output])

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
            #print('')
            #print('BEFORE')
            #print('Min ' + str(np.amin(self.prev_out)) + ', ', end='')
            #print('Max ' + str(np.amax(self.prev_out)) + ', ', end='')
            #print('Mean ' + str(np.mean(self.prev_out)) + ', ', end='')
            #print('STD ' +  str(np.std(self.prev_out )))

            layer_out = f_prev_out([self.train_data])[0]

            #layer_out = pre_process_clusters(layer_out, convolute)

            #print('AFTER')
            #print('Min ' + str(np.amin(layer_out)) + ', ', end='')
            #print('Max ' + str(np.amax(layer_out)) + ', ', end='')
            #print('Mean ' + str(np.mean(layer_out)) + ', ', end='')
            #print('STD ' +  str(np.std(layer_out )))
            #print('')

        # Save the transformed input if the flag is set.
        if self.SHOULD_SAVE_RAW and self.force_create[layer_index]:
            self.__save_raw_output(self.raw_out_loc + save_name + '.csv', layer_out)

        # If the anchor vectors should be calculated calculate them.
        if self.should_set_weights[layer_index]:
            tmp_centroids = load_or_create_centroids(self.force_create[layer_index], self.centroids_out_loc +
                save_name + '.csv', self.batch_size, layer_out, input_shape, self.subsample,
                self.filter_size, k, self.filter_params, layer_index,
                self.model_wrapper, convolute=convolute)

            if len(tmp_centroids) != k:
                output_shape = (output_shape[0], len(tmp_centroids))
                assert_shape = (len(tmp_centroids), assert_shape[1])

            if assert_shape is not None:
                assert tmp_centroids.shape == assert_shape, 'Shape is %s' % str(tmp_centroids.shape)

            tmp_centroids = tmp_centroids.reshape(output_shape)

            return tmp_centroids
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

