from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.models import load_model

from clustering import load_or_create_centroids
from clustering import build_patch_vecs
from model_wrapper import ModelWrapper

from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import pickle
import plotly.plotly as py
from plotly.tools import FigureFactory as FF
import plotly.tools as tls
from helpers.mathhelper import angle_between

def add_convlayer(model, nkern, subsample, filter_size, input_shape=None, weights=None):

    if input_shape is not None:
        convLayer = Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same', subsample=subsample, input_shape=input_shape)
    else:
        convLayer = Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same', subsample=subsample)

    model.add(convLayer)

    if not weights is None:
        params = convLayer.get_weights()
        bias = params[1]

        convLayer.set_weights([weights, bias])

    model.add(Activation('relu'))
    maxPoolingOut = MaxPooling2D(pool_size=(2,2), strides=(2,2))
    model.add(maxPoolingOut)
    convout_f = K.function([model.layers[0].input], [maxPoolingOut.output])
    return convout_f


def add_fclayer(model, output_dim, weights=None):
    denseLayer = Dense(output_dim)

    model.add(denseLayer)

    if not weights is None:
        bias = denseLayer.get_weights()[1]
        denseLayer.set_weights([weights, bias])

    fcOutLayer = Activation('relu')
    model.add(fcOutLayer)
    fcOut_f = K.function([model.layers[0].input], [fcOutLayer.output])
    return fcOut_f

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
                weights = weights.reshape(sp[1], sp[2])
                anchor_vectors.append(weights)

        return anchor_vectors


def fetch_data(test_size):
    dataset = datasets.fetch_mldata('MNIST Original')
    data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
    data = data[:, np.newaxis, :, :]

    return train_test_split(data / 255.0, dataset.target.astype('int'), test_size=test_size)


def create_model(train_percentage, should_set_weights):
    (train_data, test_data, train_labels, test_labels) = fetch_data(0.3)

    remaining = int(len(train_data) * train_percentage)

    train_labels = np_utils.to_categorical(train_labels, 10)
    test_labels = np_utils.to_categorical(test_labels, 10)

    train_data = train_data[0:remaining]
    train_labels = train_labels[0:remaining]

    print 'Running for %.2f%% test size' % (train_percentage * 100.)
    print 'The training data has a length of %i' % (len(train_data))

    input_shape = (1, 28, 28)
    subsample=(1,1)
    filter_size=(5,5)
    batch_size = 128
    nkerns = (6, 16)
    force_create = False

    input_centroids = [None] * 5

    model = Sequential()

    if should_set_weights[0]:
        print 'Setting conv layer 0 weights'
        input_centroids[0] = load_or_create_centroids(force_create, 'data/centroids/centroids0.h5', batch_size, train_data, input_shape, subsample, filter_size, nkerns[0])

    convout0_f = add_convlayer(model, nkerns[0], subsample, filter_size, input_shape=input_shape, weights=input_centroids[0])

    if should_set_weights[1]:
        print 'Setting conv layer 1 weights'
        c0Out = convout0_f([train_data])[0]
        input_shape = (nkerns[0], 14, 14)
        input_centroids[1] = load_or_create_centroids(force_create, 'data/centroids/centroids1.h5', batch_size, c0Out, input_shape, subsample, filter_size, nkerns[1])

    convout1_f = add_convlayer(model, nkerns[1], subsample, filter_size, input_shape=input_shape, weights=input_centroids[1])


    model.add(Flatten())

    if should_set_weights[2]:
        print 'Setting fc layer 0 weights'
        c1Out = convout1_f([train_data])[0]
        input_shape = (nkerns[1], 7, 7)
        input_centroids[2] = load_or_create_centroids(force_create, 'data/centroids/centroids2.h5',
                                batch_size, c1Out, input_shape, subsample,
                                filter_size, 120, convolute=False)
        sp = input_centroids[2].shape
        input_centroids[2] = input_centroids[2].reshape(sp[1], sp[0])

    fc0_f = add_fclayer(model, 120, weights=input_centroids[2])

    if should_set_weights[3]:
        print 'Setting fc layer 1 weights'
        fc0Out = fc0_f([train_data])[0]
        input_shape = (120,)
        input_centroids[3] = load_or_create_centroids(force_create, 'data/centroids/centroids3.h5',
                                    batch_size, fc0Out, input_shape, subsample,
                                    filter_size, 84, convolute=False)
        sp = input_centroids[3].shape
        input_centroids[3] = input_centroids[3].reshape(sp[1], sp[0])

    fc1_f = add_fclayer(model, 84, weights=input_centroids[3])

    if should_set_weights[4]:
        print 'Setting classifier weights'
        fc1Out = fc1_f([train_data])[0]
        input_shape=(84,)
        input_centroids[4] = load_or_create_centroids(force_create, 'data/centroids/centroids4.h5', batch_size, fc1Out, input_shape, subsample, filter_size, 10, convolute=False)
        sp = input_centroids[4].shape
        input_centroids[4] = input_centroids[4].reshape(sp[1], sp[0])

    classification_layer = Dense(10)
    model.add(classification_layer)

    if should_set_weights[4]:
        bias = classification_layer.get_weights()[1]
        classification_layer.set_weights([input_centroids[4], bias])

    model.add(Activation('softmax'))

    print 'Compiling model'
    opt = SGD(lr = 0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(train_data, train_labels, batch_size=128, nb_epoch=20, verbose=1)
    (loss, accuracy) = model.evaluate(test_data, test_labels, batch_size=128, verbose=1)

    print ''
    print 'Accuracy %.9f%%' % (accuracy * 100.)

    # print 'Saving'
    # model.save_weights(save_model_filename)
    return ModelWrapper(accuracy, input_centroids)

test_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

def run_experiment():
    base_model = create_model(1.0, [False] * 5)
    all_models = [base_model]

    for test_size in test_sizes:
        model = create_model(test_size, [True] * 5)
        all_models.append(model)

    accuracies = [model.accuracy for model in all_models]

    with open('data/models.h5', 'wb') as f:
        pickle.dump(accuracies, f)


def load_experiment():
    with open('data/credentials.txt') as f:
        creds = f.readlines()

    tls.set_credentials_file(username=creds[0].rstrip(), api_key=creds[1].rstrip())
    with open('data/models.h5', 'rb') as f:
        models = pickle.load(f)

    header_mag_row = ['Data %', 'Accuracy']
    for i in range(len(models[1].centroids[0])):
        header_mag_row.append('Centroid %i Mag' % (i))

    header_angle_row = ['Data %', 'Accuracy']
    for i in range(len(models[1].centroids[0])):
        header_angle_row.append('Centroid %i Angle' % (i))

    mag_rows = []
    angle_rows = []

    # Create a unit vector along the first dimension axis.
    comparison_vec = np.zeros(25)
    comparison_vec[0] = 1

    for i, model in enumerate(models):
        if i == 0:
            data_size_str = 'NA'
        else:
            data_size_str = '%.2f%%' % ((1. - test_sizes[i - 1]) * 100.)

        mag_row = [data_size_str, '%.2f%%' % (model.accuracy * 100.)]
        angle_row = [data_size_str, '%.2f%%' % (model.accuracy * 100.)]

        if not model.centroids[0] is None:
            for centroid in model.centroids[0]:
                centroid = centroid.flatten()
                # Get the maginitude of the centroid.
                centroid_mag = np.linalg.norm(centroid)
                mag_row.append('%.5f' % (centroid_mag))

                angle = angle_between(centroid / centroid_mag, comparison_vec)
                angle_row.append(angle)

        mag_rows.append(mag_row)
        angle_rows.append(angle_row)

    mag_data_matrix = [header_mag_row]
    mag_data_matrix.extend(mag_rows)

    angle_data_matrix = [header_angle_row]
    angle_data_matrix.extend(angle_rows)

    table = FF.create_table(mag_data_matrix)
    py.iplot(table, filename='CentroidMagnitudeComparison')

    table2 = FF.create_table(angle_data_matrix)
    py.iplot(table2, filename='CentroidAngleComparison')


# load_experiment()
run_experiment()
