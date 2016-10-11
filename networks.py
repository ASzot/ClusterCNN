from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
        model = Sequential()
        model.add(Convolution2D(20, 5, 5, border_mode='same', input_shape=(depth, height, width)))
        model.add(Activation('relu'))
        maxPoolingOut = MaxPooling2D(pool_size=(2,2), strides=(2,2))
        model.add(maxPoolingOut)

        convout0_f = theano.function([model.get_input(train=False)], maxPoolingOut.get_output(train=False))

        model.add(Convolution2D(50, 5, 5, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
