# Just used to assemble all of the data in a readable way.
from os import listdir
from os.path import isfile, join
import pickle
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
from plotly.tools import FigureFactory as FF
import plotly.tools as tls
from load_runner import LoadRunner


def set_plotly_creds():
    with open('data/credentials.txt') as f:
        creds = f.readlines()

    tls.set_credentials_file(username=creds[0].rstrip(), api_key=creds[1].rstrip())

def plot_scalar_data(data, title):
    header = ['Layer #', 'STD', 'AVG', 'MIN', 'MAX']

    for model_data in data:
        rows = [header]
        for i, sub_data in enumerate(model_data[2]):
            std = np.std(sub_data)
            avg = np.average(sub_data)
            min_val = np.amin(sub_data)
            max_val = np.amax(sub_data)
            rows.append([i, std, avg, min_val, max_val])

        table = FF.create_table(rows)
        table.layout.update({'title': '%s: %.5f%%' % (model_data[0], model_data[1])})

        py.iplot(table, filename='%s %s' % (model_data[0], title))

def plot_accuracies():
    # Draw a graph for each of the accuracies.
    load_path = 'data/accuracies/'
    train_accuracy_files = [f for f in listdir(load_path) if isfile(join(load_path, f))]

    for train_accuracy_file in train_accuracy_files:
        # Get the constant factor.
        const_fact_str = train_accuracy_file[len('train_accuracies') + 1:]
        if const_fact_str.startswith('neg'):
            const_fact_str = const_fact_str[len('neg') + 1:]

        # Remove the h5 extension.
        const_fact_str = const_fact_str[:len(const_fact_str) - 3]
        const_fact = int(const_fact_str)

        plot_accuracy(load_path + train_accuracy_file, 'Weight Factor: %i' % const_fact, 'data/figs/graph_%i.jpg' % const_fact)


def plot_accuracy(filename, title, savename):
    with open(filename) as f:
        all_train_data = pickle.load(f)

        data_used_increments = [i[0] for i in all_train_data]
        reg_accuracies = [i[1] for i in all_train_data]
        kmeans_accuracies = [i[2] for i in all_train_data]

        fig = plt.figure()
        reg_line = plt.plot(data_used_increments, reg_accuracies)
        kmeans_line = plt.plot(data_used_increments, kmeans_accuracies)
        plt.setp(reg_line, color='r', linewidth=2.0)
        plt.setp(kmeans_line, color='b', linewidth=2.0)
        fig.suptitle(title, fontsize=20)
        plt.xlabel('Train Data Used', fontsize=18)
        plt.ylabel('Accuracy', fontsize=16)
        # plt.axis([0.0, 0.25, 0.5, 0.75, 1.0])

        fig.savefig(savename)

def plot_anchor_vec_data():
    load_runner = LoadRunner(None)
    angle_data = load_runner.run_or_load('data/anchor_vecs_angles.h5')
    mag_data = load_runner.run_or_load('data/anchor_vecs_mag_reg.h5')

    set_plotly_creds()
    plot_scalar_data(angle_data, 'Angle Data')
    plot_scalar_data(mag_data, 'Magnitude Data')

plot_accuracy('data/accuracies/normalized_train_accuracies.h5', 'Batch Size=128', 'data/figs/fig0.png')
plot_accuracy('data/accuracies/small_batch_normalized_train_accuracies.h5', 'Batch Size=1', 'data/figs/fig1.png')
