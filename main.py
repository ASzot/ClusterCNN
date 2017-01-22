import matplotlib.pyplot as plt
import pickle
import numpy as np
from helpers.hyper_params import HyperParamData
from helpers.hyper_param_search import HyperParamSearch
from model_wrapper import ModelWrapper
from helpers.printhelper import PrintHelper as ph


def get_hyperparams():
    return HyperParamData(
        input_shape = (1, 28, 28),
        subsample=(1,1),
        patches_subsample = (1,1),
        filter_size=(5,5),
        batch_size = 5,
        nkerns = (6,16),
        fc_sizes = (120, 84, 10,),
        n_epochs = 10,
        min_variances = [0.3, 0.3, 4., 50., 0.6],
        selection_percentages = [0.03, 0.5, 0.0, 0.0, 0.0],
        use_filters = (True, False, False, False, False),
        activation_func = 'relu',
        extra_path = '',
        should_set_weights = [True] * 5,
        should_eval = True,
        remaining = 0)


def single_test():
    hyperparams = get_hyperparams()
    hyperparams.extra_path = 'kmeans'
    model = ModelWrapper(hyperparams, force_create=True)
    model.create_model()
    model.eval_performance()

    ph.linebreak()
    ph.disp('Anchor Vec Spread Std: ')
    ph.disp(model.anchor_vec_spreads_std)

    ph.linebreak()
    ph.disp('Layer Weight Stds: ')
    ph.disp(model.layer_weight_stds)
    ph.disp('Layer Weight Avgs: ')
    ph.disp(model.layer_weight_avgs)

    ph.linebreak()
    ph.disp('Layer Mag Avg: ')
    ph.disp(model.layer_anchor_mags_avg)
    ph.disp('Layer Mag Std: ')
    ph.disp(model.layer_anchor_mags_std)

    ph.linebreak()
    print 'Model prediction distribution: ' + str(np.std(model.pred_dist))
    print model.pred_dist
    print 'Actual distribution: ' + str(np.std(model.actual_dist))
    print model.actual_dist


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

    interval = 5
    trails = 3
    total = 750
    kmeans_accs = []
    reg_accs = []

    ph.DISP = False

    total_range = np.concatenate([np.arange(0, total, interval), np.arange(800, 5000, 100)])

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
            total_kmeans_acc += model.create_model()

            hyperparams.should_set_weights = [False] * 5
            hyperparams.extra_path = 'reg'
            model = ModelWrapper(hyperparams, force_create=False)
            total_reg_acc += model.create_model()

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
    plt.legend(handles=[kmeans_line, reg_line], loc=4)
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

