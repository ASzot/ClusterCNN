import matplotlib.pyplot as plt
from helpers.mathhelper import angle_between
import pickle
import numpy as np

def main(name, loc, fig, ax):
    with open('data/av/' + name + '0.h5') as f:
        pre_av_data = pickle.load(f)

    train_size = pre_av_data[0]
    pre_av = pre_av_data[1][0]

    with open('data/av/' + name + '980.h5') as f:
        av_data = pickle.load(f)

    av = av_data[1][0]

    angles = []
    for i in range(5):
        av[i] = np.array(av[i])
        pre_av[i] = np.array(pre_av[i])

        both_av = zip(av[i], pre_av[i])

        layer_angles = []
        for pre_train_vec, train_vec in both_av:
            angle = angle_between(pre_train_vec, train_vec)
            layer_angles.append(angle)

        angles.append(np.mean(layer_angles))


    return angles




if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    kmeans_angles = main('kmeans', 'center', fig, ax)

    reg_angles = main('reg', 'center', fig, ax)

    left, width = 0.1, 0.6
    bottom, height = 0.0, 0.0
    left_table = 0.15
    table_width = 0.8
    table_height = 0.3

    rect_main = [left, bottom, width, height]
    rect_table1 = [left_table, 0.1, table_width, table_height]
    rect_table2 = [left_table, bottom, table_width, table_height]

    axMain = plt.axes(rect_main)
    axTable1 = plt.axes(rect_table1, frameon =False)
    axTable2 = plt.axes(rect_table2, frameon =False)
    axTable1.axes.get_xaxis().set_visible(False)
    axTable2.axes.get_xaxis().set_visible(False)
    axTable1.axes.get_yaxis().set_visible(False)
    axTable2.axes.get_yaxis().set_visible(False)

    axMain.axes.get_xaxis().set_visible(False)
    axMain.axes.get_yaxis().set_visible(False)

    rowLabels = [r'k-means $\Delta \theta$', r'Random $\Delta \theta$']
    colLabels = ['C1', 'C2', 'F1', 'F2', 'F3']

    axTable1.set_title('Change in Angle Between No BP and Fully Trained with BP')
    axTable1.table(cellText=[kmeans_angles, reg_angles], loc='upper center',
                           rowLabels=rowLabels, colLabels=colLabels, cellLoc='center')
    plt.tight_layout()

    #plt.savefig('data/figs/asdf.png')
    plt.show()

