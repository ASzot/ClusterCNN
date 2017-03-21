import pickle
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
import matplotlib.pyplot as plt

def get_freq_percents(labels):
    y = np.bincount(labels)
    ii = np.nonzero(y)[0]
    return np.vstack((ii,y[ii])).T

with open('data/outputs.h5', 'rb') as f:
    outputs = pickle.load(f)

# For each compute distance to cluster center.
dists = []
colors = []
right_cnt = 0
wrong_cnt = 0

pred_labels = []

for output in outputs:
    t_sample = output[5]
    centroid = output[6]
    is_correct = output[0]
    if is_correct:
        colors.append('b')
        right_cnt += 1
    else:
        colors.append('r')
        wrong_cnt += 1

    centroid = np.array(centroid)
    t_sample = np.array(t_sample)
    dist = cosine_dist(t_sample, centroid)
    pred_labels.append(output[3])

    dists.append(dist)


pred_labels = list(get_freq_percents(pred_labels))
pred_labels = sorted(pred_labels, key=lambda x: x[1], reverse=True)
print(pred_labels)
print('With an agreement of %.2f' % (float(right_cnt) / float(right_cnt +
    wrong_cnt)))

plt.scatter(dists, [0.0] * len(dists), c=colors)

plt.show()


