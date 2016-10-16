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
    print 'Running for %.2f%% test size' % (test_size * 100.)

    return train_test_split(data / 255.0, dataset.target.astype('int'), test_size=test_size)


def create_model(test_size, shouldSetWeights):
    (trainData, testData, trainLabels, testLabels) = fetch_data(test_size)

    print 'The training data has a length of %i' % (len(trainData))

    trainLabels = np_utils.to_categorical(trainLabels, 10)
    testLabels = np_utils.to_categorical(testLabels, 10)

    input_shape = (1, 28, 28)
    subsample=(1,1)
    filter_size=(5,5)
    batch_size = 128
    nkerns = (6, 16)
    forceCreate = True

    inputCentroids = [None] * 5

    model = Sequential()

    if shouldSetWeights[0]:
        print 'Setting conv layer 0 weights'
        inputCentroids[0] = load_or_create_centroids(forceCreate, 'data/centroids/centroids0.h5', batch_size, trainData, input_shape, subsample, filter_size, nkerns[0])

    convout0_f = add_convlayer(model, nkerns[0], subsample, filter_size, input_shape=input_shape, weights=inputCentroids[0])

    if shouldSetWeights[1]:
        print 'Setting conv layer 1 weights'
        c0Out = convout0_f([trainData])[0]
        input_shape = (nkerns[0], 14, 14)
        inputCentroids[1] = load_or_create_centroids(forceCreate, 'data/centroids/centroids1.h5', batch_size, c0Out, input_shape, subsample, filter_size, nkerns[1])

    convout1_f = add_convlayer(model, nkerns[1], subsample, filter_size, input_shape=input_shape, weights=inputCentroids[1])


    model.add(Flatten())

    if shouldSetWeights[2]:
        print 'Setting fc layer 0 weights'
        c1Out = convout1_f([trainData])[0]
        input_shape = (nkerns[1], 7, 7)
        inputCentroids[2] = load_or_create_centroids(forceCreate, 'data/centroids/centroids2.h5', batch_size, c1Out, input_shape, subsample, filter_size, 120, convolute=False)
        sp = inputCentroids[2].shape
        inputCentroids[2] = inputCentroids[2].reshape(sp[1], sp[0])

    fc0_f = add_fclayer(model, 120, weights=inputCentroids[2])

    if shouldSetWeights[3]:
        print 'Setting fc layer 1 weights'
        fc0Out = fc0_f([trainData])[0]
        input_shape = (120,)
        inputCentroids[3] = load_or_create_centroids(forceCreate, 'data/centroids/centroids3.h5', batch_size, fc0Out, input_shape, subsample, filter_size, 84, convolute=False)
        sp = inputCentroids[3].shape
        inputCentroids[3] = inputCentroids[3].reshape(sp[1], sp[0])

    fc1_f = add_fclayer(model, 84, weights=inputCentroids[3])

    if shouldSetWeights[4]:
        print 'Setting classifier weights'
        fc1Out = fc1_f([trainData])[0]
        input_shape=(84,)
        inputCentroids[4] = load_or_create_centroids(forceCreate, 'data/centroids/centroids4.h5', batch_size, fc1Out, input_shape, subsample, filter_size, 10, convolute=False)
        sp = inputCentroids[4].shape
        inputCentroids[4] = inputCentroids[4].reshape(sp[1], sp[0])

    classificationLayer = Dense(10)
    model.add(classificationLayer)

    if shouldSetWeights[4]:
        bias = classificationLayer.get_weights()[1]
        classificationLayer.set_weights([inputCentroids[4], bias])

    model.add(Activation('softmax'))

    print 'Compiling model'
    opt = SGD(lr = 0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # model.fit(trainData, trainLabels, batch_size=128, nb_epoch=20, verbose=1)
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
    print 'Accuracy %.9f%%' % (accuracy * 100.)

    # print 'Saving'
    # model.save_weights(save_model_filename)
    return ModelWrapper(accuracy, inputCentroids)


test_sizes = []
test_size = 0.90
while test_size > 0.3:
    test_sizes.append(test_size)
    test_size -= 0.1

def run_experiment():
    base_model = create_model(0.3, [False] * 5)
    all_models = [base_model]

    for test_size in test_sizes:
        model = create_model(test_size, [True] * 5)
        all_models.append(model)

    with open('data/models.h5', 'wb') as f:
        pickle.dump(all_models, f)

def load_experiment():
    with open('data/credentials.txt') as f:
        creds = f.readlines()

    tls.set_credentials_file(username=creds[0], api_key=creds[1])
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
            data_size_str = '%.2f%%' % (test_sizes[i - 1] * 100.)

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


load_experiment()
