import theano
import timeit
from theano import tensor as T
from theano.tensor.nnet import conv2d
import numpy
import pickle
from model_builder import build_lenet_model
from data_helper import load_data
from model_builder import build_lenet_layers

from logistic_regression import LogisticRegression
from hidden_layer import HiddenLayer


def save_model(layers, savePath='model.pickle'):
    with open(savePath, 'wb') as f:
        pickle.dump([param.get_value() for layer in layers for param in layer.params ], f, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(loadPath='model.pickle'):
    with open(loadPath, 'rb') as f:
        layers = []
        pair = []
        count = 0
        for param in pickle.load(f):
            if count == 2:
                layers.append(pair)
                pair = []
                count = 0
            pair.append(param)
            count += 1
        layers.append(pair)
        return layers

def train_lenet5(datasets, batchSize = 500, nEpochs=200):
    """ Implements LeNet
    :type nkerns: list of ints
    :param nkerns: number of conv kernels for the ith layer
    """

    # Load the data.
    rng = numpy.random.RandomState()

    trainSetX, trainSetY = datasets[0]
    validateSetX, validateSetY = datasets[1]
    testSetX, testSetY = datasets[2]

    # Get the number of batches for each data set.
    nTrainBatches = datasets[0][0].get_value(borrow=True).shape[0] // batchSize
    nValidateBatches = datasets[1][0].get_value(borrow=True).shape[0] // batchSize
    nTestBatches = datasets[2][0].get_value(borrow=True).shape[0] // batchSize

    # Index to the batch
    index = T.lscalar()

    models = build_lenet_model(batchIndex = index,
                        batchSize=batchSize,
                        rng = rng,
                        trainSet = datasets[0],
                        validateSet = datasets[1],
                        testSet = datasets[2])
    trainModel, validateModel, testModel, layers = models

    startTime = timeit.default_timer()

    epoch = 0
    doneLooping = False

    while (epoch < nEpochs) and (not doneLooping):
        epoch = epoch + 1

        print 'Training @ epoch', epoch
        currentTime = timeit.default_timer()
        print '%.2f has passed' % (currentTime - startTime)

        for miniBatchIndex in range(nTrainBatches):
            iter = (epoch - 1) * nTrainBatches + miniBatchIndex

            if iter % 100 == 0:
                print 'Training @ iter = ', iter

            cost_ij = models[0](miniBatchIndex)

    endTime = timeit.default_timer()
    print 'Optimization complete.'
    print 'Ran for %.2f' % (endTime - startTime)

    # Run the validation set.
    validationLosses = [
        validateModel(i)
        for i in range(nValidateBatches)
    ]

    validationScore = numpy.mean(validationLosses)

    print 'Validation score %f %%' % (validationScore * 100.)

    return layers

def create_model():
    datasets = load_data('mnist.pkl')
    model = train_lenet5(nEpochs = 1, datasets = datasets)

    print '- Saving model'
    save_model(layers=model)

def create_from_file(batchSize = 500):
    layers = load_model()
    datasets = load_data('mnist.pkl')
    testSetX, testSetY = datasets[2]
    nTestBatches = testSetX.get_value(borrow=True).shape[0] // batchSize

    index = T.lscalar()

    # The image data.
    x = T.matrix('x')
    # The label data. This is an integer vector
    y = T.ivector('y')

    rng = numpy.random.RandomState()

    layers = build_lenet_layers(batchSize=batchSize,
                                rng = rng,
                                x = x,
                                layers = layers)

    outputLayer = layers[-1]

    conv1 = layers[0]
    weights = conv1.W.get_value(borrow=True)
    bias = conv1.b.get_value(borrow=True)
    print weights.shape
    print weights[0][0][0]

    # testModel = theano.function(
    #     [index],
    #     outputLayer.errors(y),
    #     givens = {
    #         x: testSetX[index * batchSize: (index + 1) * batchSize],
    #         y: testSetY[index * batchSize: (index + 1) * batchSize]
    #     }
    # )
    #
    # testLosses = [
    #     testModel(i)
    #     for i in range(nTestBatches)
    # ]
    #
    # testScore = numpy.mean(testLosses)
    #
    # print '%.2f%%' % (testScore * 100.)
