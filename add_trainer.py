from model_analyzer import ModelAnalyzer
import os
import matplotlib.pyplot as plt
import numpy as np
import uuid


class AddTrainer(object):
    def __init__(self, model):
        self.model = model

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
            for x,y in matching_samples_xy:
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



