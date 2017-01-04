

def plot_table(stat_data, name):
    cols = ['mean', 'std', 'var', 'min', 'max']
    rows = ['Layer %i' % i for i in range(len(stat_data))]

    cell_text = []
    for layer_stat in stat_data:
        cell_text.append(['%.2f' % stat for stat in layer_stat])

    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.table(cellText=cell_text, rowLabels=rows, colLabels=cols, loc='center')
    fig.savefig('data/' + name + '.png')
    plt.close(fig)


def create_models():
    trails = 2
    reg_acc_total = 0.0
    kmeans_acc_total = 0.0

    for i in range(trails):
        kmeans_model = create_model(0.1, [True] * 5, extra_path='kmeans')
        reg_model = create_model(0.1, [False] * 5, extra_path='reg')
        kmeans_acc_total += kmeans_model.accuracy
        reg_acc_total += reg_model.accuracy

    reg_acc_avg = reg_acc_total / float(trails)
    kmeans_acc_avg = kmeans_acc_total / float(trails)

    print 'Reg accuracy %.2f' % (reg_acc_avg * 100.)
    print 'Kmeans accuracy %.2f' % (kmeans_acc_avg * 100.)


def test_accuracy():
    kmeans_model = create_model(0.2, [True] * 5, extra_path='whitened_cosine')
    print 'Accuracy obtained was ' + str(kmeans_model.accuracy)


def create_accuracies():
    all_accuracies = []
    for use_data in np.arange(0.0, 0.4, 0.05):
        kmeans_model = create_model(use_data, [True] * 5, extra_path='kmeans_train')
        kmeans_accuracy = kmeans_model.accuracy
        ph.disp('Accuracy for kmeans %.9f%%' % (kmeans_accuracy), ph.HEADER)
        del kmeans_model

        reg_model = create_model(use_data, [False] * 5, extra_path='reg_train')
        reg_accuracy = reg_model.accuracy
        ph.disp('Accuracy for regular %.9f%%' % (reg_accuracy), ph.HEADER)
        del reg_model

        all_accuracies.append((kmeans_accuracy, reg_accuracy))

    with open('data/accuracies/accuracy_comparison.dat', 'wb') as f:
        pickle.dump(all_accuracies, f)


def analyze_accuracies():
    with open('data/accuracies/accuracy_comparison.dat', 'rb') as f:
        all_accuracies = pickle.load(f)

    kmeans_accuracies, reg_accuracies = zip(*all_accuracies)

    model_analyzer = ModelAnalyzer()
    model_analyzer.plot_data(kmeans_accuracies, 'Kmeans accuracy', 'g')
    model_analyzer.plot_data(reg_accuracies, 'Reg accuracy', 'r')
    model_analyzer.show_table()


def analyze_models():
    model_analyzer = ModelAnalyzer()

    if not model_analyzer.load('data/centroids/pythonreg'):
        print 'Could not load reg models'
        return False

    reg_data_means = model_analyzer.get_data_means()
    reg_data_stds = model_analyzer.get_data_stds()
    print reg_data_means


    if not model_analyzer.load('data/centroids/python_kmeans_TEST_sub_mean_norm'):
        print 'Could not load models.'
        return False

    # model_analyzer.whiten_data(reg_data_stds)
    kmeans_data_means = model_analyzer.get_data_means()
    kmeans_data_stds = model_analyzer.get_data_stds()
    print kmeans_data_means

    model_analyzer.plot_data(reg_data_stds, 'Reg std', 'g')
    model_analyzer.plot_data(kmeans_data_stds, 'Kmeans std', 'y')
    model_analyzer.plot_data(kmeans_data_means, 'Kmeans mean', 'r')
    model_analyzer.plot_data(reg_data_means, 'Reg mean', 'b')

    model_analyzer.show_table()


# create_accuracies()
# analyze_accuracies()
create_models()
# analyze_models()
# test_accuracy()
