import numpy as np
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as MultiCoreTSNE
from sklearn.manifold import TSNE
import sklearn.preprocessing as preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity
from helpers.printhelper import PrintHelper as ph
import random
import pickle


def subtract_mean(cluster_vec):
    return cluster_vec - np.mean(cluster_vec)


def plot_samples(samples, anchor_vecs, labels):
    ph.disp('Performing TSNE')

    tsne_use_samples = 2000
    tsne_plot_samples = 500
    ndim = 2
    post_normalize = False

    samples = samples[0:tsne_use_samples]
    labels = labels[0:tsne_use_samples]

    if ndim == 2:
        tsne_model = MultiCoreTSNE(n_jobs=5)
    else:
        tsne_model = TSNE(n_components=3, verbose=True)

    flattened_x = [np.array(sample).flatten() for sample in samples]
    samples = np.array(samples)

    all_data = []
    all_data.extend(samples)
    all_data.extend(anchor_vecs)

    all_data = np.array(all_data, dtype='float64')

    # normalize all of the input vectors.
    #all_data = preprocessing.normalize(all_data)

    transformed_all_data = tsne_model.fit_transform(all_data)

    with open('data/vis_data/asdf.h5', 'wb') as f:
        pickle.dump(transformed_all_data, f)

    #with open('data/vis_data/asdf.h5', 'rb') as f:
    #    transformed_all_data = pickle.load(f)

    av_count = len(anchor_vecs)
    if post_normalize:
        transformed_all_data = preprocessing.normalize(transformed_all_data)
    vis_data = transformed_all_data[:-av_count]
    plot_avs = transformed_all_data[-av_count:]

    ph.disp('Done fitting data')

    print('The shape is ' + str(vis_data.shape))
    all_data = random.sample(list(vis_data), tsne_plot_samples)

    ph.disp('There are %i samples to plot' % len(vis_data))

    all_data = np.array(all_data)
    all_data = list(zip(all_data, labels))

    # Pair each label with a color.
    colors = ['red', 'blue', 'green', 'yellow', 'SaddleBrown', 'black',
                'MediumTurquoise', 'OrangeRed', 'Violet', 'white']

    le = preprocessing.LabelEncoder()
    le.fit(labels)

    if ndim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        use_plt = ax
    elif ndim == 2:
        use_plt = plt
    else:
        raise ValueError('Invalid number of dimensions')

    colors_index = 0
    colors_map = []
    for dataClass in le.classes_:
        sampleOfClass = [sample for sample, label in all_data if label == dataClass]
        sampleOfClass = np.array(sampleOfClass)

        plot_dimensions = []
        for dim in range(ndim):
            plot_dimensions.append(sampleOfClass[:,dim])

        use_plt.scatter(*plot_dimensions, c = colors[colors_index],
                marker='o')

        colors_map.append(colors[colors_index])

        colors_index += 1
        if (colors_index == len(colors)):
            colors_index = 0

    ph.disp('Plotting all anchor vectors')

    for i, av in enumerate(plot_avs):
        t_vals = np.linspace(0, 1, 2)

        av_comps = []
        for dim in range(ndim):
            av_comps.append(av[dim] * t_vals)

        use_plt.plot(*av_comps, linewidth=2.0, color=colors_map[i])

    print('There are {} anchor vectors'.format(len(plot_avs)))

    plt.show()



def convert_index_to_onehot(indicies, num_classes):
    for index in indicies:
        vec = np.zeros(num_classes)
        vec[index] = 1.0
        yield vec

def convert_onehot_to_index(vectors):
    indicies = []
    for i in range(len(vectors)):
        start_count = len(indicies)

        for j, vec_ele in enumerate(vectors[i]):
            if vec_ele != 0:
                indicies.append(j)

        if len(indicies) == start_count:
            raise ValueError('New elements have not been appended.')
    return indicies


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_layer_anchor_vectors(layer_data):
    sp = layer_data.shape
    if len(sp) > 2:
        anchor_vecs = []
        for conv_filter in layer_data:
            conv_filter = conv_filter.flatten()
            anchor_vecs.append(conv_filter)
        return anchor_vecs
    else:
        print(sp[0])
        return layer_data


def get_biases(model0):
    for layer in model0.model.layers:
        params = layer.get_weights()
        if len(params) > 0:
            biases = params[1]
            yield biases


def unset_bias(model0):
    for layer in model0.model.layers:
        params = layer.get_weights()
        if len(params) > 0:
            weights = params[0]
            zero_bias = np.zeros(params[1].shape)
            layer.set_weights([weights, zero_bias])


# A helper function to get the anchor vectors of all layers..
def get_anchor_vectors(model0):
    anchor_vectors = []

    for layer in model0.model.layers:
        params = layer.get_weights()
        if len(params) > 0:
            weights = params[0]
            if len(weights.shape) > 2:
                # This is a convolution layer
                add_anchor_vectors = []
                for conv_filter in weights:
                    conv_filter = conv_filter.flatten()
                    add_anchor_vectors.append(conv_filter)
                anchor_vectors.append(add_anchor_vectors)
            else:
                sp = weights.shape
                weights = weights.reshape(sp[1], sp[0])
                anchor_vectors.append(weights)

    return anchor_vectors


def get_anchor_vector_angles(layer_anchor_vecs):
    angles = []
    for anchor_vecs in layer_anchor_vecs:
        layer_angles = []
        for anchor_vec in anchor_vecs:
            compare_vec = np.zeros(len(anchor_vec))
            compare_vec[0] = 1.
            angle = angle_between(compare_vec, anchor_vec)
            layer_angles.append(angle)
        angles.append(layer_angles)

    return angles


def set_anchor_vectors(model, anchor_vectors, nkerns, filter_size):
    sps = [anchor_vector.shape for anchor_vector in anchor_vectors]

    # Conolutional layer 0.
    anchor_vectors[0] = anchor_vectors[0].reshape(sps[0][0], 1, filter_size[0], filter_size[1])

    # Convolutional layer 1.
    sp = anchor_vectors[1].shape
    anchor_vectors[1] = anchor_vectors[1].reshape(sps[1][0], nkerns[0], filter_size[0], filter_size[1])

    # Switch the dimensions of the FC layers.
    for i in range(2, 5):
        anchor_vectors[i] = anchor_vectors[i].reshape(sps[i][1], sps[i][0])

    anchor_vectors_index = 0
    for i, layer in enumerate(model.layers):
        params = layer.get_weights()
        if len(params) > 0:
            # This is a layer that has network parameters.
            set_anchor_vector = anchor_vectors[anchor_vectors_index]
            anchor_vectors_index += 1
            weights = params[0]
            bias = params[1]
            assert set_anchor_vector.shape == weights.shape, 'Anchor Vec Shape: %s, Weights Shape: %s' % (set_anchor_vector.shape, weights.shape)
            # Does not matter if it is a convolution or fully connected layer.
            model.layers[i].set_weights([set_anchor_vector, bias])
