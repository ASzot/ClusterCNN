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

def add_convlayer(model, nkern, subsample, filter_size, input_shape=None, weights=None):

    if input_shape is not None:
        convLayer = Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same', subsample=subsample, input_shape=input_shape)
    else:
        convLayer = Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same', subsample=subsample)

    model.add(convLayer)

    if not weights is None:
        params = convLayer.get_weights()
        bias = params[1]
        startWeights = params[0]

        convLayer.set_weights([weights, bias])
        afterWeights = convLayer.get_weights()[0]

        # Get the difference between the two.
        sp = weights.shape
        prod = 1
        for dim in sp:
            prod *= dim

        startWeights = startWeights.reshape(prod)
        afterWeights = afterWeights.reshape(prod)


        distance = np.linalg.norm(startWeights - afterWeights)
        print 'distance is %.9f' % (distance)

        raise ValueError()


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

def get_weight_angles(model0):
    for layer in model0.layers:
        print layer

def fetch_data(test_size):
    dataset = datasets.fetch_mldata('MNIST Original')
    data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
    data = data[:, np.newaxis, :, :]
    print 'Running for %.2f%% test size' % (test_size * 100.)

    (trainData, testData, trainLabels, testLabels) = train_test_split(data / 255.0, dataset.target.astype('int'), test_size=test_size)


def run_experiment(test_size, shouldSetWeights):
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

    input0Centroids = None
    input1Centroids = None
    input2Centroids = None
    input3Centroids = None
    input4Centroids = None

    model = Sequential()

    if shouldSetWeights[0]:
        print 'Setting conv layer 0 weights'
        input0Centroids = load_or_create_centroids(forceCreate, 'data/centroids/centroids0.h5', batch_size, trainData, input_shape, subsample, filter_size, nkerns[0])

    convout0_f = add_convlayer(model, nkerns[0], subsample, filter_size, input_shape=input_shape, weights=input0Centroids)

    if shouldSetWeights[1]:
        print 'Setting conv layer 1 weights'
        c0Out = convout0_f([trainData])[0]
        input_shape = (nkerns[0], 14, 14)
        input1Centroids = load_or_create_centroids(forceCreate, 'data/centroids/centroids1.h5', batch_size, c0Out, input_shape, subsample, filter_size, nkerns[1])

    convout1_f = add_convlayer(model, nkerns[1], subsample, filter_size, input_shape=input_shape, weights=input1Centroids)


    model.add(Flatten())

    if shouldSetWeights[2]:
        print 'Setting fc layer 0 weights'
        c1Out = convout1_f([trainData])[0]
        input_shape = (nkerns[1], 7, 7)
        input2Centroids = load_or_create_centroids(forceCreate, 'data/centroids/centroids2.h5', batch_size, c1Out, input_shape, subsample, filter_size, 120, convolute=False)
        sp = input2Centroids.shape
        input2Centroids = input2Centroids.reshape(sp[1], sp[0])

    fc0_f = add_fclayer(model, 120, weights=input2Centroids)

    if shouldSetWeights[3]:
        print 'Setting fc layer 1 weights'
        fc0Out = fc0_f([trainData])[0]
        input_shape = (120,)
        input3Centroids = load_or_create_centroids(forceCreate, 'data/centroids/centroids3.h5', batch_size, fc0Out, input_shape, subsample, filter_size, 84, convolute=False)
        sp = input3Centroids.shape
        input3Centroids = input3Centroids.reshape(sp[1], sp[0])

    fc1_f = add_fclayer(model, 84, weights=input3Centroids)

    if shouldSetWeights[4]:
        print 'Setting classifier weights'
        fc1Out = fc1_f([trainData])[0]
        input_shape=(84,)
        input4Centroids = load_or_create_centroids(forceCreate, 'data/centroids/centroids4.h5', batch_size, fc1Out, input_shape, subsample, filter_size, 10, convolute=False)
        sp = input4Centroids.shape
        input4Centroids = input4Centroids.reshape(sp[1], sp[0])

    classificationLayer = Dense(10)
    model.add(classificationLayer)

    if shouldSetWeights[4]:
        bias = classificationLayer.get_weights()[1]
        classificationLayer.set_weights([input4Centroids, bias])

    model.add(Activation('softmax'))

    print 'Compiling model'
    opt = SGD(lr = 0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # model.fit(trainData, trainLabels, batch_size=128, nb_epoch=20, verbose=1)
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
    print 'Accuracy %.9f%%' % (accuracy * 100.)

    # print 'Saving'
    # model.save_weights(save_model_filename)
    return ModelWrapper(model, accuracy)

#
model0 = run_experiment(0.5, [True] * 5)
model1 = run_experiment(0.9, [False] * 5)

print 'M0 had an accuracy of %.9f' % (model0.accuracy * 100.)
# print 'M1 had an accuracy of %.9f' % (model1.accuracy * 100.)



# for i in range(1):
#     filename = '/model' + str(i) + '.h5'
#     print filename
#     model = load_model(filename)
#     get_weight_angles(model)
