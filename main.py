import matplotlib.pyplot as plt
import pickle
from helpers.hyper_params import HyperParamData


def plot_accuracies():
    with open('data/kmeans_accuracies.h5', 'r') as f:
        kmeans_acc_data = pickle.load(f)

    with open('data/reg_accuracies.h5', 'r') as f:
        reg_accuracies = pickle.load(f)

    X = [point * 10 for point in range(len(reg_accuracies))]

    plt.plot(X, kmeans_acc_data, color='blue')
    plt.plot(X, reg_accuracies, color='red')

    plt.show()


def run():
    hyperparams = HyperParamData(
        input_shape = (1, 28, 28),
        subsample=(1,1),
        patches_subsample = (1,1),
        filter_size=(5,5),
        batch_size = 5,
        nkerns = (6,16),
        fc_sizes = (120, 84, 10,),
        n_epochs = 10,
        min_variances = (0.3, 0.3, 4., 50., 0.6),
        selection_percentages = (0.03, 0.5, 0.0, 0.0, 0.0),
        use_filters = (True, False, False, False, False),
        activation_func = 'relu',
        extra_path = '',
        should_set_weights = [True] * 5,
        remaining = 100)
