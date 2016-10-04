# API imports
import theano
import timeit
from theano import tensor as T
from theano.tensor.nnet import conv2d
import numpy as np
import pickle
import warnings

# User defined imports
from model_builder import build_lenet_layers
from kmeans import OnlineKMeans
from kmeans import save_centroids
from kmeans import load_centroids
from lenet import get_image_patches
from logistic_regression import LogisticRegression
from hidden_layer import HiddenLayer
from sklearn.cluster import MiniBatchKMeans


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

        addPatchVecs = [patch.reshape(patch.shape[0] * patch.shape[0]) for patch in patches]
        patchVecs.extend(addPatchVecs)
    return patchVecs

def construct_centroids(batchSize, trainSetX, inputShape, stride, filterShape, nClusters):
    print '- Building centroids'
    patchVecs = build_patch_vecs(batchSize, trainSetX, inputShape, stride, filterShape)
    batchScalingFactor = len(patchVecs) // len(trainSetX)
    # Remember there will be many more batches with the patches
    mbk = MiniBatchKMeans(init='k-means++',
                        n_clusters=nClusters,
                        batch_size=batchSize * batchScalingFactor,
                        max_no_improvement=10,
                        reassignment_ratio=0.01,
                        verbose=True)

    print 'Fitting mbk'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mbk.fit(patchVecs)
    print 'Finished fitting'

    centroids = mbk.cluster_centers_
    print '- Finished building centroids'
    return centroids


def load_or_create_centroids(forceCreate, filename, batchSize, inpData, inputShape, stride, filterShape, nkern):
    if not forceCreate:
        try:
            centroids = load_centroids(filename)
        except IOError:
            forceCreate = True

    if forceCreate:
        centroids = construct_centroids(batchSize, inpData, inputShape, stride, filterShape, nkern)
        save_centroids(centroids, filename)

    return centroids


def create_pretrained(datasets, filterShape, stride, inputShape, nkerns, rng, batchSize, x, y, batchIndex, forceCreate):
    sharedTrainSetX, sharedTrainSetY = datasets[0]
    trainSetX = sharedTrainSetX.get_value(borrow=True)

    centroids = load_or_create_centroids(forceCreate, 'centroids/centroids0.h5', batchSize, trainSetX, inputShape, stride, filterShape, nkerns[0])
    sp = centroids.shape
    centroids = centroids.reshape(sp[0], filterShape[0], filterShape[1])
    sp = centroids.shape
    centroids = centroids.reshape(sp[0], 1, sp[1], sp[2])

    nTrainBatches = trainSetX.shape[0] // batchSize

    layers = build_lenet_layers(rng, batchSize, None, x, inputShape, nkerns, filterShape)
    layer0 = layers[0]

    # Set the weights of this filter.
    print 'Setting weights of first conv/pooling layer to centroids'

    layer0.W.set_value(centroids, borrow=True)

    # Compute the output from this layer.
    runLayer0 = theano.function(
        [batchIndex],
        layer0.output,
        givens = {
            x: sharedTrainSetX[batchIndex * batchSize: (batchIndex + 1) * batchSize]
        }
    )

    # Calculate the output.
    outputLayer0 = [
        runLayer0(i)
        for i in range(nTrainBatches)
    ]

    outputLayer0 = np.array(outputLayer0)

    # Rearrange so the filter count is first.
    sp = outputLayer0.shape

    outputLayer0 = outputLayer0.reshape(sp[2], sp[0], sp[1], sp[3], sp[4])
    # Flatten out the batches.
    flattenedOutput = []

    filterFlattenedOutputs = []
    for kernel in outputLayer0:
        filterFlattenedOutput = []
        for batch in kernel:
            for subBatch in batch:
                patchVec = []
                for ele in subBatch:
                    for row in ele:
                        patchVec.append(row)
                filterFlattenedOutput.append(patchVec)

        filterFlattenedOutputs.append(filterFlattenedOutput)


    filterFlattenedOutputs = np.array(filterFlattenedOutputs)
    print filterFlattenedOutputs.shape
    allCentroidsLayer0 = []

    newInputShape = [((inputShape[i] - filterShape[i]+1) / 2) for i in range(2)]
    # Do for every filter.
    for i, flattenedOutputs in enumerate(filterFlattenedOutputs):
        filename = 'centroids/centroidsL0_%i.h5' % (i)
        centroidsLayer0 = load_or_create_centroids(forceCreate, filename, batchSize, flattenedOutputs, newInputShape, stride, filterShape, nkerns[1])
        # Convert back to the matrix form.
        sp = centroidsLayer0.shape
        centroidsLayer0 = centroidsLayer0.reshape(sp[0], filterShape[0], filterShape[1])
        allCentroidsLayer0.append(centroidsLayer0)

    allCentroidsLayer0 = np.array(allCentroidsLayer0)
    sp = allCentroidsLayer0.shape
    allCentroidsLayer0 = allCentroidsLayer0.reshape(sp[1], sp[0], sp[2], sp[3])

    print 'Setting weights of second conv/pooling layer to centroids'
    layers[1].W.set_value(allCentroidsLayer0, borrow=True)

    return layers
