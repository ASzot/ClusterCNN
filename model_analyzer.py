import csv
import numpy as np
import matplotlib.pyplot as plt

from helpers.mathhelper import svd_whiten
from scipy.cluster.vq import whiten

class ModelAnalyzer(object):
    def __init__(self):
        self.layer_names = ['c0', 'c1', 'f0', 'f1', 'f2']
        self.layer_anchor_vecs = None


    def load(self, data_loc):
        self.layer_anchor_vecs = []
        for layer_name in self.layer_names:
            anchor_vecs = []
            try:
                with open(data_loc + '/cluster/' + layer_name + '.csv', 'r') as f:
                    csvreader = csv.reader(f, delimiter=',')
                    for row in csvreader:
                        anchor_vecs.append(row)
            except IOError:
                return False
            self.layer_anchor_vecs.append(np.array(anchor_vecs).astype(np.float))
        return True


    def get_data_means(self):
        # Get the mean of each layer
        layer_means = []
        for anchor_vecs in self.layer_anchor_vecs:
            # Get the mean of this layer.
            # Collapse the anchor_vec.
            anchor_vecs = np.array(anchor_vecs)
            anchor_vecs = anchor_vecs.flatten()
            layer_mean = np.mean(anchor_vecs)
            layer_means.append(layer_mean)
        return layer_means


    def get_data_stds(self):
        layer_stds = []
        for anchor_vecs in self.layer_anchor_vecs:
            anchor_vecs = anchor_vecs.flatten()
            layer_std = np.std(anchor_vecs)
            layer_stds.append(layer_std)
        return layer_stds


    def whiten_data(self, match_stds):
        # Subtract the mean of the data.
        for i, anchor_vecs in enumerate(self.layer_anchor_vecs):
            anchor_vecs *= match_stds[i]

            self.layer_anchor_vecs[i] = anchor_vecs


    def plot_data(self, plt_data, color):
        assert len(plt_data) == 5, 'Invalid number of layers in the data'
        plt.plot(range(5), plt_data, color)


    def show_table(self):
        plt.show()
