import pickle
import theano
import numpy
import os
from theano import tensor as T

def load_data(nUseSamples, datasetPath='mnist.pkl', notShared=False):
    # Check in the current folder to see if the file exists.
    if (not os.path.isfile(datasetPath)):
        return None

    print '- Loading data'

    with open(datasetPath, 'rb') as f:
        try:
            trainSet, validSet, testSet = pickle.load(f)
        except:
            return None

    trainSet = (trainSet[0][0:nUseSamples], trainSet[1][0:nUseSamples])
    validSet = (validSet[0][0:nUseSamples], validSet[1][0:nUseSamples])
    testSet = (testSet[0][0:nUseSamples], testSet[1][0:nUseSamples])

    if notShared:
        return [trainSet, validSet, testSet]

    def shared_dataset(dataXY):
        dataX, dataY = dataXY
        sharedX = theano.shared(numpy.asarray(dataX, dtype=theano.config.floatX), borrow=True)
        sharedY = theano.shared(numpy.asarray(dataY, dtype=theano.config.floatX), borrow=True)

        return sharedX, T.cast(sharedY, 'int32')

    testSetX, testSetY = shared_dataset(testSet)
    validSetX , validSetY = shared_dataset(validSet)
    trainSetX, trainSetY = shared_dataset(trainSet)

    print '- Done loading'

    return [(trainSetX, trainSetY), (validSetX, validSetY), (testSetX, testSetY)]
