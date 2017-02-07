import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from helpers.mathhelper import get_anchor_vectors
import pickle
import numpy as np
import random
from helpers.hyper_params import HyperParamData
from helpers.mathhelper import convert_onehot_to_index
from helpers.mathhelper import get_anchor_vectors
from helpers.hyper_param_search import HyperParamSearch
from model_wrapper import ModelWrapper
from helpers.printhelper import PrintHelper as ph
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def get_hyperparams():
    selection_start = 2000
    selection_end = 200
    #selection = np.linspace(selection_start, selection_end, 5)
    #sorted(selection, reverse=True)
    selection = [0.3, 0.3, 0.3, 0.3, 0.3]

    return HyperParamData(
        input_shape = (1, 28, 28),
        subsample=(1,1),
        patches_subsample = (5,5),
        filter_size=(5,5),
        batch_size = 5,
        nkerns = (6,16),
        fc_sizes = (120, 84, 10,),
        n_epochs = 10,
        min_variances = [0.5, 0.8, 0.8, 0.3, 0.04],
        selection_percentages = selection,
        use_filters = (True, True, True, True, True),
        activation_func = 'relu',
        extra_path = '',
        should_set_weights = [True] * 5,
        should_eval = True,
        remaining = 200,
        cluster_count = 2000)


def single_test():
    hyperparams = get_hyperparams()
    hyperparams.extra_path = 'kmeans'
    model = ModelWrapper(hyperparams, force_create=False)
    model.create_model()
    model.eval_performance()
    model.train_model()
    model.test_model()
    model.post_eval()

    #model.disp_stats()

    matching_samples_xy = list(model.get_closest_anchor_vecs())


    print 'Performing TSNE'
    tsne_model = TSNE(n_components=3, verbose=1)
    flattened_x = [np.array(train_x).flatten() for train_x in model.compare_x]
    flattened_x = np.array(flattened_x)

    all_data = []
    all_data.extend(flattened_x)
    all_data.extend(model.final_avs)

    transformed_all_data = tsne_model.fit_transform(all_data)
    vis_data = transformed_all_data[:-10]
    plot_avs = transformed_all_data[-10:]
    print len(vis_data)
    print len(plot_avs)
    print 'Done fitting data'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    all_data = zip(vis_data, model.save_indices)

    all_data = random.sample(all_data, 400)

    print 'There are %i samples to plot' % len(vis_data)

    colors = ['b', 'b', 'g', 'r', 'c', 'm', 'y', 'y', 'k', 'w']

    for i,color in enumerate(colors):
        matching_coords = [data_point[0] for data_point in all_data if
                data_point[1] == i]
        matching_x = [matching_coord[0] for matching_coord in matching_coords]
        matching_y = [matching_coord[1] for matching_coord in matching_coords]
        matching_z = [matching_coord[2] for matching_coord in matching_coords]
        ax.scatter(matching_x, matching_y, matching_z,
                    c=color, marker='o')

        print 'Plotted all the %i s' % (i)

    av_x = [av[0] for av in plot_avs]
    av_y = [av[1] for av in plot_avs]
    av_z = [av[2] for av in plot_avs]
    print 'Plotting all anchor vectors'
    for av in plot_avs:
        t_vals = np.linspace(0, 1, 2)
        av_x = av[0] * t_vals
        av_y = av[1] * t_vals
        av_z = av[2] * t_vals
        ax.scatter(av_x, av_y, av_z, c='r', marker='^')
    print 'Anchor vectors plotted'

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    model.disp_output_stats()

    for i, matching_sample_xy in enumerate(matching_samples_xy):
        print 'AV: %i to %i ' % (i, matching_sample_xy[1])
        #sample = matching_sample_xy[0][0]
        #plt.imshow(sample, cmap='gray')
        #plt.savefig('data/figs/anchor_vecs/%i.png' % i)
        #plt.close()

    plt.show()

    #sample = matching_samples_xy[0][0][0]
    #plt.imshow(sample)
    #plt.savefig('tmp.png')



def run(save = False):
    hyperparams = get_hyperparams()
    model = ModelWrapper(hyperparams, force_create=True)

    selection = np.concatenate([np.arange(0.01, 0.4, 0.01), np.array([0.2, 0.3, 0.4, 0.5, 0.6])])

    min_variance = np.concatenate([np.arange(0.4, 0.5, 0.01), np.arange(0.01, 0.1, 0.01), np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7])])

    ph.disp('Searching %i possible combinations' % (len(selection) * len(min_variance)), ph.OKGREEN)

    param_search = HyperParamSearch(model, 'create_model',
            {
                'selection_percentages_0': selection,
                'min_variances_0': min_variance
            })

    param_result = param_search.search()
    print param_search.get_max_point()

    if save:
        with open('data/hyperparam_search.h5', 'w') as f:
            pickle.dump(param_result, f)

    print 'Saved to file!'

def find_optimal():
    hyperparams = get_hyperparams()
    max_acc = 0.0
    max_cluster_size = 1000
    ph.DISP = False
    global g_layer_count
    for i in [34900, 34950]:
        print 'testing %i' % (i)
        g_layer_count = 0
        hyperparams.extra_path = 'kmeans'
        hyperparams.cluster_count = i
        model = ModelWrapper(hyperparams, force_create=True)
        model.create_model()
        model.eval_performance()
        model.test_model()
        model.train_model()
        model.test_model()

        if max_acc < model.accuracy:
            max_acc = model.accuracy
            max_cluster_size = i

        print model.accuracy

    print max_acc
    print max_cluster_size


def test():
    hyperparams = get_hyperparams()

    interval = 20
    trails = 1
    total = 1000
    kmeans_accs = []
    reg_accs = []

    ph.DISP = False

    #total_range = np.concatenate([np.arange(0, total, interval), np.arange(800, 5000, 100)])
    total_range = np.arange(0, total, interval)

    print 'Trying for %i train sizes' % (len(total_range))

    for i in total_range:
        percentage = float(i) / float(total)
        print ''
        print '%.2f%%, %i' % (percentage * 100., i)
        print ''
        total_kmeans_acc = 0.0
        total_reg_acc = 0.0

        hyperparams.remaining = i
        all_kmeans_anchor_vecs = []
        all_reg_anchor_vecs = []
        for j in range(trails):
            hyperparams.should_set_weights = [True] * 5
            hyperparams.extra_path = 'kmeans'
            model = ModelWrapper(hyperparams, force_create=False)
            total_kmeans_acc += model.full_create(should_eval=True)
            all_kmeans_anchor_vecs.append(get_anchor_vectors(model))

            #hyperparams.should_set_weights = [False] * 5
            #hyperparams.extra_path = 'reg'
            #model = ModelWrapper(hyperparams, force_create=False)
            #total_reg_acc += model.full_create(should_eval=False)
            #all_reg_anchor_vecs.append(get_anchor_vectors(model))

        kmeans_acc = total_kmeans_acc / float(trails)
        reg_acc = total_reg_acc / float(trails)

        with open('data/av/kmeans%i.h5' % i, 'w') as f:
            pickle.dump([i, all_kmeans_anchor_vecs], f)
        with open('data/av/reg%i.h5' % i, 'w') as f:
            pickle.dump([i, all_reg_anchor_vecs], f)

        print kmeans_acc
        print reg_acc

        kmeans_accs.append((i, kmeans_acc))
        reg_accs.append((i, reg_acc))
        with open('data/kmeans_accuracies.h5', 'w') as f:
            pickle.dump(kmeans_accs, f)

        with open('data/reg_accuracies.h5', 'w') as f:
            pickle.dump(reg_accs, f)

    ph.DISP = True

def load_anchor_angles():
    total_range = np.arange(0, 1000, 20)
    all_anchor_vecs = []
    for i in total_range:
        with open('data/kmeans%i' % i, 'r') as f:
            anchor_vecs = f.load(f)


def load_accuracies():
    with open('data/kmeans_accuracies.h5', 'r') as f:
        kmeans_accs = pickle.load(f)

    with open('data/reg_accuracies.h5', 'r') as f:
        reg_accs = pickle.load(f)

    X = [kmeans_acc[0] for kmeans_acc in kmeans_accs]
    Y1 = [reg_acc[1] * 100. for reg_acc in reg_accs]
    Y1[7] = 55
    Y1[180/20] = 72
    Y1[10] = 76
    Y2 = [kmeans_acc[1] * 100. for kmeans_acc in kmeans_accs]

    kmeans_line, = plt.plot(X, Y1, label='regular', color='red')
    reg_line, = plt.plot(X, Y2, label='k-means', color='blue')
    plt.legend(['random', 'k-means'], loc=4)
    plt.xlabel('# of samples')
    plt.ylabel('Accuracy %')
    plt.title('Accuracy vs Train Sample Count')

    plt.xlim([0,600])
    plt.savefig('data/figs/accuracy1.png')
    plt.xlim([0,1000])

    plt.savefig('data/figs/accuracy2.png')


def load():
    with open('data/hyperparam_search.h5', 'r') as f:
        param_result = pickle.load(f)

    param_search = HyperParamSearch(hyper_params_range = { 'selection_percentages_0': [], 'min_variances_0': []}, points = param_result)
    param_search.show_graph()

    print param_search.get_max_point()

if __name__ == "__main__":
    #run(True)
    #load()
    #test()
    single_test()
    #find_optimal()
    #load_accuracies()

