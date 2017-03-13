import pickle
import matplotlib.pyplot as plt
import numpy as np
from os import listdir

def load_cluster(filename, index, disp=False):
    try:
        with open('data/cluster_data/%i/%s.h5' % (index, filename), 'rb') as f:
            cluster_data = pickle.load(f)
    except IOError:
        print('Could not open file ' + filename)

    percent = filename.split('_')[1]

    ret_data = {}
    ret_data['cluster']  = cluster_data[0]
    ret_data['labels']   = cluster_data[1]
    ret_data['samples']  = cluster_data[2]
    ret_data['centroid'] = cluster_data[3]
    ret_data['percent'] = int(percent)

    cluster_var = np.var(ret_data['cluster'])

    ret_data['var'] = cluster_var
    ret_data['real_var'] = np.var(ret_data['samples'])

    if disp:
        print('For a concetration of %s' % percent)
        print('Cluster has %i samples' % (len(ret_data['cluster'])))
        print('Cluster has a variance of %.7f' % (cluster_var))
        print('Cluster has a real variance of %.7f' % (np.var(ret_data['samples'])))
        print('')

    return ret_data


def analyze_single():
    load_cluster('1_29', 1, True)
    load_cluster('2_97', 1, True)


def analyze_all():
    all_vars = []
    all_percents = []
    sizes = []

    for filename in listdir('data/cluster_data/1/'):
        ret_data = load_cluster(filename.split('.')[0], 1)
        all_vars.append(ret_data['real_var'])
        all_percents.append(ret_data['percent'])
        sizes.append(2 * len(ret_data['cluster']))

    vars_min = np.amin(all_vars)
    vars_max = np.amax(all_vars)

    axes = plt.gca()
    plt.scatter(all_percents, all_vars, s=sizes)
    axes.set_ylim([vars_min,vars_max])
    plt.show()

analyze_single()
