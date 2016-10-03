# API imports
import theano
import timeit
from theano import tensor as T
from theano.tensor.nnet import conv2d
import numpy as np
import pickle

# User defined imports
from model_builder import build_lenet_layers
from kmeans import k_means
from kmeans import save_centroids
from kmeans import load_centroids
from lenet import get_image_patches
from logistic_regression import LogisticRegression
from hidden_layer import HiddenLayer


def save_model(layers, savePath):
    with open(savePath, 'wb') as f:
        pickle.dump([param.get_value() for layer in layers for param in layer.params ], f, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(loadPath, rng, batchSize, x, inputShape, nkerns, filterShape):
    with open(loadPath, 'rb') as f:
        layersWeights = []
        pair = []
        count = 0
        for param in pickle.load(f):
            if count == 2:
                layersWeights.append(pair)
                pair = []
                count = 0
            pair.append(param)
            count += 1
        layersWeights.append(pair)

    return build_lenet_layers(rng, batchSize, layersWeights, x, inputShape, nkerns, filterShape)


def train_lenet5(trainSet, trainModel, layers, rng, batchSize, batchIndex, nEpochs=200):
    trainSetX, trainSetY = trainSet

    # Get the number of batches for each data set.
    nTrainBatches = trainSetX.get_value(borrow=True).shape[0] // batchSize

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

            cost_ij = trainModel(miniBatchIndex)

    endTime = timeit.default_timer()
    print 'Optimization complete.'
    print 'Ran for %.2f' % (endTime - startTime)

    return layers


def build_patch_vecs(batchSize, trainSetX, inputShape, stride, filterShape):
    patchVecs = []
    total = len(trainSetX)
    for i, trainX in enumerate(trainSetX):
        if i % (total // 10) == 0:
            print '%.2f%%' % ((float(i) / float(total)) * 100.)
        patches = get_image_patches(trainX, inputShape, stride, filterShape)
        for patch in patches:
            patchVec = [col for row in patch for col in row]
            patchVecs.append(patchVec)
    return patchVecs

def construct_centroids(batchSize, trainSetX, inputShape, stride, filterShape, nClusters):
    print '- Building centroids'
    patchVecs = build_patch_vecs(batchSize, trainSetX, inputShape, stride, filterShape)
    centroids = k_means(patchVecs, nClusters)
    print '- Finished building centroids'
    return centroids


def load_or_create_centroids(forceCreate, filename, batchSize, trainSetX, inputShape, stride, filterShape, nkern):

    if not forceCreate:
        try:
            centroids = load_centroids(filename)
        except IOError:
            forceCreate = True

    if forceCreate:
        centroids = construct_centroids(batchSize, trainSetX, inputShape, stride, filterShape, nkern)
        save_centroids(centroids, filename)

    return centroids


def create_pretrained(datasets, filterShape, stride, inputShape, nkerns, rng, batchSize, x, y, batchIndex):
    sharedTrainSetX, sharedTrainSetY = datasets[0]
    trainSetX = sharedTrainSetX.get_value(borrow=True)

    centroids = load_or_create_centroids(False, 'centroids/centroids0.h5', batchSize, trainSetX, inputShape, stride, filterShape, nkerns[0])

    nTrainBatches = trainSetX.shape[0] // batchSize

    layers = build_lenet_layers(rng, batchSize, None, x, inputShape, nkerns, filterShape)
    layer0 = layers[0]

    # Set the weights of this filter.
    print 'Setting weights of first conv/pooling layer to centroids'
    layer0.W = centroids

    # Compute the output from this layer.
    runLayer0 = theano.function(
        [batchIndex],
        layer0.output,
        givens = {
            x: sharedTrainSetX[batchIndex * batchSize: (batchIndex + 1) * batchSize]
        }
    )

    outputLayer0 = [
        runLayer0(i)
        for i in range(nTrainBatches)
    ]

    outputLayer0 = np.array(outputLayer0)
    # Flatten out the batches.
    flattenedOutput = []
    for batch in outputLayer0:
        for subBatch in batch:
            patchVec = []
            for kernel in subBatch:
                for ele in kernel:
                    for row in ele:
                        patchVec.append(row)
            flattenedOutput.append(patchVec)


    flattenedOutput = np.array(flattenedOutput)

    centroidsLayer0 = load_or_create_centroids(False, 'centroids/centroids1.h5', batchSize, trainSetX, inputShape, stride, filterShape, nkerns[1])

    layer1 = layers[1]
    print 'Setting weights of second conv/pooling layer to centroids'
    layer1.W = centroidsLayer0

    layers = (layer0, layer1, layers[2], layers[3])

    return layers
