import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.vq import whiten

import pickle
import random

from helpers.mathhelper import *
from helpers.printhelper import PrintHelper as ph

from MulticoreTSNE import MulticoreTSNE as TSNE
import sklearn.preprocessing as preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity

from model_wrapper import ModelWrapper


class ModelAnalyzer(ModelWrapper):
    """
    Just to reduce clutter in model_wrapper.py
    This is to intended to hold a lot of the visualization
    functionality of the model to try to get a sense of
    what is going on in the network.
    """

    def perform_tsne(self):
        matching_samples_xy = list(self.get_closest_anchor_vecs())

        ph.disp('Performing TSNE')
        dims = 2

        tsne_model = TSNE(n_jobs=6)
        flattened_x = [np.array(train_x).flatten() for train_x in self.compare_x]
        flattened_x = np.array(flattened_x)

        all_data = []
        all_data.extend(flattened_x)
        all_data.extend(self.final_avs)

        all_data = np.array(all_data, dtype='float64')

        # normalize all of the input vectors.
        all_data = preprocessing.normalize(all_data)

        transformed_all_data = tsne_model.fit_transform(all_data)
        with open('data/vis_data/tsne.h5', 'wb') as f:
            pickle.dump(transformed_all_data, f)

        #try:
        #    raise IOError()
        #    with open('data/vis_data/tsne.h5', 'rb') as f:
        #        transformed_all_data = pickle.load(f)
        #except IOError:

        #transformed_all_data = preprocessing.normalize(transformed_all_data, norm='l2')

        vis_data = transformed_all_data[:-10]
        plot_avs = transformed_all_data[-10:]

        ph.disp('Done fitting data')

        if dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # This relies on the assumption that after t-sne the data elements
        # are still in the same order.
        all_data = list(zip(vis_data, self.save_indices))

        all_data = random.sample(all_data, 200)

        ph.disp('There are %i samples to plot' % len(vis_data))


        # 0 - red
        # 1 - blue
        # 2 - green
        # 3 - yellow
        # 4 - brown
        # 5 - black
        # 6 - cyan
        # 7 - orange
        # 8 - pink
        # 9 - white
        colors = ['red', 'blue', 'green', 'yellow', 'SaddleBrown', 'black',
                'MediumTurquoise', 'OrangeRed', 'Violet', 'white']

        for i, color in enumerate(colors):
            matching_coords = [data_point[0] for data_point in all_data if
                    data_point[1] == i]
            matching_x = [matching_coord[0] for matching_coord in matching_coords]
            matching_y = [matching_coord[1] for matching_coord in matching_coords]

            if dims == 3:
                matching_z = [matching_coord[2] for matching_coord in matching_coords]
                ax.scatter(matching_x, matching_y, matching_z,
                            c=color, marker='o')
            elif dims == 2:
                plt.scatter(matching_x, matching_y, c=color, marker='o')
            else:
                raise ValueError('Invalid number of dimensions')

            ph.disp('Plotted all the %i s' % (i))

        ph.disp('Plotting all anchor vectors')

        cur_index = 0
        for av in plot_avs:
            t_vals = np.linspace(0, 1, 2)
            av_x = av[0] * t_vals
            av_y = av[1] * t_vals
            use_color = colors[cur_index]

            if dims == 3:
                av_z = av[2] * t_vals
                ax.plot(av_x, av_y, av_z, linewidth=2.0, color = use_color,
                        label='AV %i' % cur_index)
            elif dims == 2:
                plt.plot(av_x, av_y, linewidth=2.0, color=use_color)
            else:
                raise ValueError('Invalid number of dimensions')
            cur_index += 1

        ph.disp('Anchor vectors plotted')

        if dims == 3:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        self.disp_output_stats()

        for i, matching_sample_xy in enumerate(matching_samples_xy):
            ph.disp('AV: %i to %i ' % (i, matching_sample_xy[1]))

            #sample = matching_sample_xy[0][0]
            #plt.imshow(sample, cmap='gray')
            #plt.savefig('data/figs/anchor_vecs/%i.png' % i)
            #plt.close()

        plt.show()

        #sample = matching_samples_xy[0][0][0]
        #plt.imshow(sample)
        #plt.savefig('tmp.png')



    def disp_stats(self):
        ph.linebreak()
        ph.disp('Layer Bias Std', ph.OKBLUE)
        ph.disp(self.layer_bias_stds)
        ph.disp('Layer Bias Avg', ph.OKBLUE)
        ph.disp(self.layer_bias_avgs)

        ph.linebreak()
        ph.disp('Anchor Vec Spread Std: ', ph.OKBLUE)
        ph.disp(self.anchor_vec_spreads_std)
        ph.disp('Anchor Vec Spread Avg: ', ph.OKBLUE)
        ph.disp(self.anchor_vec_spreads_avg)

        ph.linebreak()
        ph.disp('Layer Weight Stds: ', ph.OKBLUE)
        ph.disp(self.layer_weight_stds)
        ph.disp('Layer Weight Avgs: ', ph.OKBLUE)
        ph.disp(self.layer_weight_avgs)

        ph.linebreak()
        ph.disp('Layer Mag Avg: ', ph.OKBLUE)
        ph.disp(self.layer_anchor_mags_avg)
        ph.disp('Layer Mag Std: ', ph.OKBLUE)
        ph.disp(self.layer_anchor_mags_std)

        ph.linebreak()
        pred_dist_std = np.std(self.pred_dist)
        actual_dist_std = np.std(self.actual_dist)
        ph.disp('Model prediction distribution: ' + str(pred_dist_std), ph.FAIL)
        ph.disp(self.pred_dist)
        ph.disp('Model prediction map', ph.FAIL)
        ph.disp(self.pred_to_actual)
        ph.disp('Actual distribution: ' + str(actual_dist_std), ph.FAIL)
        ph.disp(self.actual_dist)
        dist_ratio = pred_dist_std / actual_dist_std
        ph.disp('Distribution Ratio: ' + str(dist_ratio), ph.FAIL)


    def disp_output_stats(self):
        final_avs = get_anchor_vectors(self)[-1]

        #final_avs = [np.linalg.norm(final_av) for final_av in final_avs]

        per_anchor_vec_avg = [np.mean(final_av) for final_av in final_avs]
        per_anchor_vec_std = [np.std(final_av) for final_av in final_avs]

        ph.disp(per_anchor_vec_avg)
        ph.disp(per_anchor_vec_std)


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


    def post_eval(self):
        """
        Collects statistics about each anchor vector.

        :returns: Nothing
        """
        all_anchor_vecs = get_anchor_vectors(self)
        all_biases = get_biases(self)
        for anchor_vecs in all_anchor_vecs:
            self.__set_layer_stats(anchor_vecs)

        # If you care about the bias information for some reason
        #for biases in all_biases:
        #    self.__set_layer_bias_stats(biases)


    def __set_layer_bias_stats(self, biases):
        biases = np.array(list(biases))

        self.layer_bias_stds.append(np.std(biases))
        self.layer_bias_avgs.append(np.mean(biases))


    def __set_layer_stats(self, anchor_vecs):
        layer_std = np.std(anchor_vecs)
        layer_avg = np.mean(anchor_vecs)

        anchor_mags = [np.linalg.norm(anchor_vec) for anchor_vec in anchor_vecs]
        anchor_mag_std = np.std(anchor_mags)
        anchor_mag_avg = np.mean(anchor_mags)

        anchor_vec_spreads = []
        for i, anchor_vec in enumerate(anchor_vecs):
            compare_angles = []
            for j, compare_vec in enumerate(anchor_vecs):
                if j == i:
                    continue
                angle = angle_between(compare_vec, anchor_vec)
                angle *= (180.0 / np.pi)
                compare_angles.append(angle)
            compare_angle_avg = np.mean(compare_angles)
            anchor_vec_spreads.append(compare_angle_avg)

        anchor_vec_spread_avg = np.mean(np.mean(anchor_vec_spreads))
        anchor_vec_spread_std = np.std(np.std(anchor_vec_spreads))

        self.anchor_vec_spreads_std.append(anchor_vec_spread_std)
        self.anchor_vec_spreads_avg.append(anchor_vec_spread_avg)

        self.layer_anchor_mags_avg.append(anchor_mag_avg)
        self.layer_anchor_mags_std.append(anchor_mag_std)

        self.layer_weight_stds.append(layer_std)
        self.layer_weight_avgs.append(layer_avg)


