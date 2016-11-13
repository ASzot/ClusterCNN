import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.vq import whiten

from helpers.mathhelper import get_layer_anchor_vectors


class ModelAnalyzer(object):
    def __init__(self):
        self.layer_names = ['c0', 'c1', 'f0', 'f1', 'f2']
        self.layer_anchor_vecs = None
        self.all_raw_data = None
        self.line_refs = []


    def load(self, data_loc, cluster=True):
        self.layer_anchor_vecs = []

        if cluster:
            add_part = '/cluster/'
        else:
            add_part = '/raw/'

        for layer_name in self.layer_names:
            anchor_vecs = []
            try:
                with open(data_loc + add_part + layer_name + '.csv', 'r') as f:
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


    def get_data_mags(self):
        all_layer_mags = []
        for anchor_vecs in self.layer_anchor_vecs:
            layer_mags = []
            for anchor_vec in anchor_vecs:
                mag = np.linalg.norm(anchor_vec)
                layer_mags.append(mag)
            all_layer_mags.append(layer_mags)
        return all_layer_mags


    def get_mag_stat(self):
        mags = self.get_data_mags()
        means = []
        for layer_mags in mags:
            mean_layer_mag = np.mean(layer_mags)
            std_layer_mag = np.std(layer_mags)
            means.append((mean_layer_mag, std_layer_mag))

        return means


    def whiten_data(self, match_mags):
        # Subtract the mean of the data.
        for i, anchor_vecs in enumerate(self.layer_anchor_vecs):
            # anchor_vecs *= match_stds[i]
            # anchor_vecs = whiten(anchor_vecs)
            layer_match_mags = match_mags[i]
            for j, match_mag in enumerate(layer_match_mags):
                anchor_vecs[j] = (anchor_vecs[j] / np.linalg.norm(anchor_vecs[j]))

            anchor_vecs = np.array(anchor_vecs)
            # anchor_vecs -= np.mean(anchor_vecs)
            self.layer_anchor_vecs[i] = anchor_vecs


    def plot_data(self, plt_data, title, color):
        handle, = plt.plot(range(len(plt_data)), plt_data, color, label=title)
        self.line_refs.append(handle)


    def show_table(self):
        # plt.legend()
        plt.legend(bbox_to_anchor=(0., 1.0, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        # plt.legend(handles=self.line_refs, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
