import matplotlib.pyplot as plt
import pickle
import numpy as np
from helpers.hyper_params import HyperParamData
from helpers.hyper_param_search import HyperParamSearch
from model_wrapper import ModelWrapper
from helpers.printhelper import PrintHelper as ph


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
        remaining = 0,
        cluster_count = 1000)


def single_test():
    hyperparams = get_hyperparams()
    hyperparams.extra_path = 'kmeans'
    model = ModelWrapper(hyperparams, force_create=True)
    model.create_model()
    model.eval_performance()
    model.test_model()
    model.train_model()
    model.test_model()

    #ph.linebreak()
    #ph.disp('Layer Bias Std', ph.OKBLUE)
    #ph.disp(model.layer_bias_stds)
    #ph.disp('Layer Bias Avg', ph.OKBLUE)
    #ph.disp(model.layer_bias_avgs)

    ph.linebreak()
    ph.disp('Anchor Vec Spread Std: ', ph.OKBLUE)
    ph.disp(model.anchor_vec_spreads_std)
    ph.disp('Anchor Vec Spread Avg: ', ph.OKBLUE)
    ph.disp(model.anchor_vec_spreads_avg)

    ph.linebreak()
    ph.disp('Layer Weight Stds: ', ph.OKBLUE)
    ph.disp(model.layer_weight_stds)
    ph.disp('Layer Weight Avgs: ', ph.OKBLUE)
    ph.disp(model.layer_weight_avgs)

    ph.linebreak()
    ph.disp('Layer Mag Avg: ', ph.OKBLUE)
    ph.disp(model.layer_anchor_mags_avg)
    ph.disp('Layer Mag Std: ', ph.OKBLUE)
    ph.disp(model.layer_anchor_mags_std)

    ph.linebreak()
    pred_dist_std = np.std(model.pred_dist)
    actual_dist_std = np.std(model.actual_dist)
    ph.disp('Model prediction distribution: ' + str(pred_dist_std), ph.FAIL)
    ph.disp(model.pred_dist)
    ph.disp('Model prediction map', ph.FAIL)
    ph.disp(model.pred_to_actual)
    ph.disp('Actual distribution: ' + str(actual_dist_std), ph.FAIL)
    ph.disp(model.actual_dist)
    dist_ratio = pred_dist_std / actual_dist_std
    ph.disp('Distribution Ratio: ' + str(dist_ratio), ph.FAIL)


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


def test():
    hyperparams = get_hyperparams()

    interval = 20
    trails = 1
    total = 700
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
        for j in range(trails):
            hyperparams.should_set_weights = [True] * 5
            hyperparams.extra_path = 'kmeans'
            model = ModelWrapper(hyperparams, force_create=False)
            total_kmeans_acc += model.full_create()

            hyperparams.should_set_weights = [False] * 5
            hyperparams.extra_path = 'reg'
            model = ModelWrapper(hyperparams, force_create=False)
            total_reg_acc += model.full_create()

        kmeans_acc = total_kmeans_acc / float(trails)
        reg_acc = total_reg_acc / float(trails)

        print kmeans_acc
        print reg_acc

        kmeans_accs.append((i, kmeans_acc))
        reg_accs.append((i, reg_acc))

    ph.DISP = True

    with open('data/kmeans_accuracies.h5', 'w') as f:
        pickle.dump(kmeans_accs, f)

    with open('data/reg_accuracies.h5', 'w') as f:
        pickle.dump(reg_accs, f)



def load_accuracies():
    with open('data/kmeans_accuracies.h5', 'r') as f:
        kmeans_accs = pickle.load(f)

    with open('data/reg_accuracies.h5', 'r') as f:
        reg_accs = pickle.load(f)

    X = [kmeans_acc[0] for kmeans_acc in kmeans_accs]
    Y1 = [reg_acc[1] * 100. for reg_acc in reg_accs]
    Y2 = [kmeans_acc[1] * 100. for kmeans_acc in kmeans_accs]

    kmeans_line, = plt.plot(X, Y1, label='Regular', color='red')
    reg_line, = plt.plot(X, Y2, label='KMeans', color='blue')
    plt.legend([kmeans_line, reg_line], loc=4)
    plt.xlabel('# of samples')
    plt.ylabel('Accuracy %')
    plt.title('Accuracy vs Sample Count')

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
    #load_accuracies()

