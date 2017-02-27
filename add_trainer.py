from model_analyzer import ModelAnalyzer
import os
import matplotlib.pyplot as plt
import numpy as np
import uuid
from collections import defaultdict
import operator

class AddTrainer(object):
    def __init__(self, model):
        self.model = model


    def disp_clusters(self):
        sample_closest_anchor_vec = self.model.get_closest_anchor_vecs_for_samples()
        av_matching_samples = {}
        for x, y, av_i in sample_closest_anchor_vec:
            if av_i in av_matching_samples:
                av_matching_samples[av_i][y] += 1
            else:
                av_matching_samples[av_i] = defaultdict(int)

        for av_i in sorted(av_matching_samples):
            dict_to_sort = dict(av_matching_samples[av_i])
            matching_samples = sorted(dict_to_sort.items(), key=operator.itemgetter(1), reverse=True)
            total = 0

            for y, freq in matching_samples:
                total += freq

            print('Cluster %i has a total of %i samples' % (av_i, total))

            disp_count = 2
            for i in range(disp_count):
                if i >= len(matching_samples):
                    break
                if len(matching_samples[i]) != 0:
                    out_of_total = float(matching_samples[i][1]) / float(total) * 100.
                    print("--%.2f%% %i's" % (out_of_total, matching_samples[i][0]))


    def identify_clusters(self):
        k = 2
        av_matching_samples_xy = self.model.get_closest_anchor_vecs(k)

        matching_samples_count = 0
        for i, matching_samples_xy in enumerate(av_matching_samples_xy):
            matching_samples_count += 1
            data_dir = 'data/matching_avs/av%i/' % (i)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            sample_count = 0
            for x, y in matching_samples_xy:
                x = x.reshape(x.shape[1], x.shape[2])
                plt.imshow(x, cmap='gray')
                plt.savefig(data_dir + str(sample_count) + '.png')
                plt.clf()
                sample_count += 1

        self.assigned_classes = []
        for i in range(matching_samples_count):
            votes = {}
            for j in range(k):
                sample_type = int(input('Please identify sample %i for AV %i: ' % (j, i)))
                if sample_type not in votes:
                    votes[sample_type] = 1
                else:
                    votes[sample_type] += 1

            assigned_class = max(votes, key=votes.get)
            self.assigned_classes.append(assigned_class)



