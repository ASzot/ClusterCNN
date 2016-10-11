from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

from clustering import load_or_create_centroids

from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

weightsPath = None

def add_convlayer(model, nkern, subsample, filter_size, input_shape=None, weights=None):
    if input_shape is not None:
        model.add(Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same', subsample=subsample, input_shape=input_shape, weights=weights))
    else:
        model.add(Convolution2D(nkern, filter_size[0], filter_size[1], border_mode='same', subsample=subsample, weights=weights))
    model.add(Activation('relu'))
    maxPoolingOut = MaxPooling2D(pool_size=(2,2), strides=(2,2))
    model.add(maxPoolingOut)
    convout_f = K.function([model.layers[0].input], [maxPoolingOut.output])
    return convout_f


def add_fclayer(model, output_dim):
    model.add(Flatten())
    model.add(Dense(120))
    fcOutLayer = Activation('relu')
    model.add(fcOutLayer)
    fcOut_f = K.function([model.layers[0].input], [fcOutLayer.output])
    return fcOut_f



print 'Downloading MNIST'
dataset = datasets.fetch_mldata('MNIST Original')

data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
data = data[:, np.newaxis, :, :]
(trainData, testData, trainLabels, testLabels) = train_test_split(data / 255.0, dataset.target.astype('int'), test_size=0.95)

trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

print 'Compiling model'
opt = SGD(lr = 0.01)

input_shape = (1, 28, 28)
subsample=(1,1)
filter_size=(5,5)
batch_size = 128
nkerns = (6, 16)

model = Sequential()
input0Centroids = load_or_create_centroids(False, 'data/centroids/centroids0.h5', batch_size, trainData, input_shape, subsample, filter_size, nkerns[0])
bias0 = np.zeros([nkerns[0]])
convout0_f = add_convlayer(model, nkerns[0], subsample, filter_size, input_shape=input_shape, weights=[input0Centroids, bias0])

c0Out = convout0_f([trainData])[0]

input_shape = (nkerns[0], 14, 14)
input1Centroids = load_or_create_centroids(False, 'data/centroids/centroids1.h5', batch_size, trainData, input_shape, subsample, filter_size, nkerns[1])
bias1 = np.constant(0.1, shape=[nkerns[1]])
convout1_f = add_convlayer(model, nkerns[1], subsample, filter_size, input_shape=input_shape, weights=[input1Centroids, bias1])

c1Out = convout1_f([trainData])[0]



raise ValueError()





convout1_f = add_convlayer(model, 16)
model.add(Flatten())
fc0_f = add_fclayer(model, 120)
fc1_f = add_fclayer(model, 84)

model.add(Dense(10))
model.add(Activation('softmax'))


# if weightsPath is not None:
#     model.load_weights(weightsPath)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print 'Training'
# model.fit(trainData, trainLabels, batch_size=128, nb_epoch=20, verbose=1)

print 'Conv0'
c0Out = convout0_f([testData])[0]
print c0Out.shape
print 'Conv1'
c1Out = convout1_f([testData])[0]
print c1Out.shape
print 'FC0'
fc0Out = fc0Out_f([testData])[0]
print fc0Out.shape
print 'FC1'
fc1Out = fc1Out_f([testData])[0]
print fc1Out.shape
# (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
# print 'Accuracy %.9f%%' % (accuracy * 100.)

print 'saving'
# model.save_weights('saveweights.h5', overwrite=True)
