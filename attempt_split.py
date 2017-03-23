import pickle
import warnings
import sklearn.preprocessing as preprocessing

from clustering import kmeans
import numpy as np


def get_freq_percents(labels):
    y = np.bincount(labels)
    ii = np.nonzero(y)[0]
    return np.vstack((ii,y[ii])).T


def get_sorted_freq(lbls):
    lbls = list(get_freq_percents(lbls))
    return sorted(lbls, key=lambda x: x[1], reverse=True)

def pre_proc(samples):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #samples = preprocessing.scale(samples)
        samples = preprocessing.normalize(samples, norm='l2')
    return samples


def recur_cluster(samples, sample_labels, right, wrong, depth = 0):
    if depth == 1:
        return

    samples = np.array(samples)
    samples = pre_proc(samples)
    k = 10
    pre_txt = '---' * depth
    centroids, labels = kmeans(samples, k, 128, metric='sp', pre_txt = pre_txt)

    for i in range(k):
        this_real_labels = []
        recur_this_cluster = []
        for j, label in enumerate(labels):
            if label == i:
                this_real_labels.append(sample_labels[j])
                recur_this_cluster.append(samples[j])

        recur_lbls_freq = get_sorted_freq(this_real_labels)

        right.append(recur_lbls_freq[0][1])
        wrong.append(sum([lbl[1] for lbl in recur_lbls_freq[1:]]))

        print(pre_txt + '(' + str(np.var(recur_this_cluster)) + ')' + str(recur_lbls_freq))

        recur_cluster(recur_this_cluster, sample_labels, right, wrong, depth + 1)
    print('')


with open('data/tmp.h5', 'rb') as f:
    (this_cluster, real_labels) = pickle.load(f)

print('')
label_freqs = get_sorted_freq(real_labels)
print(label_freqs)

right = []
wrong = []
right.append(label_freqs[0][1])
wrong.append(sum([lbl[1] for lbl in label_freqs[1:]]))

print('%.2f' % (float(sum(right)) / float(sum(right) + sum(wrong))))
print('-' * 200)

right = []
wrong = []
recur_cluster(this_cluster, real_labels, right, wrong)
print('%.2f' % (float(sum(right)) / float(sum(right) + sum(wrong))))



