import matplotlib.pyplot as plt
import pickle
import numpy as np
from helpers.hyper_params import HyperParamData
from helpers.hyper_param_search import HyperParamSearch
from model_wrapper import ModelWrapper
from helpers.printhelper import PrintHelper as ph

def plot_accuracies():
    with open('data/kmeans_accuracies.h5', 'r') as f:
        kmeans_acc_data = pickle.load(f)

    with open('data/reg_accuracies.h5', 'r') as f:
        reg_accuracies = pickle.load(f)

    X = [point * 10 for point in range(len(reg_accuracies))]

    plt.plot(X, kmeans_acc_data, color='blue')
    plt.plot(X, reg_accuracies, color='red')

    plt.show()


def run(save = False):
    hyperparams = HyperParamData(
        input_shape = (1, 28, 28),
        subsample=(1,1),
        patches_subsample = (5,5),
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
        remaining = 100)

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


def load():
    with open('data/hyperparam_search.h5', 'r') as f:
        param_result = pickle.load(f)

    print len(param_result)
    param_search = HyperParamSearch(hyper_params_range = { 'selection_percentages_0': [], 'min_variances_0': []}, points = param_result)
    param_search.show_graph()

if __name__ == "__main__":
    #run(True)
    load()

