# API imports
import copy
import numpy as np

# User defined imports
from experiment import *
from data_helper import load_data
from model_builder import build_lenet_layers
from model_builder import build_lenet_model
from core.mathhelper import angle_between

filterShape = (5,5)
stride = (1,1)
inputShape = (28,28)
nkerns = (6, 16)
rng = np.random.RandomState()
batchSize = 500
nUseSamples = 5000
learningRate = 0.1
forceCreate = True

batchIndex = T.lscalar()
x = T.matrix('x')
y = T.ivector('y')

datasets = load_data(nUseSamples, 'mnist.pkl')

try:
    # Maybe not the best way to do things...
    if forceCreate:
        raise IOError()

    preTrainedLayers = load_model('models/pretrained.h5', rng, batchSize, x, inputShape, nkerns, filterShape)
except IOError:
    preTrainedLayers = create_pretrained(datasets, filterShape, stride, inputShape, nkerns, rng, batchSize, x, y, batchIndex, forceCreate)
    save_model(preTrainedLayers, 'models/pretrained.h5')

# Copy the array so comparisons can be later made.
preTrainedParams = []
for preTrainedLayer in preTrainedLayers:
    W = copy.deepcopy(preTrainedLayer.W.get_value(borrow=False))
    b = copy.deepcopy(preTrainedLayer.b.get_value(borrow=False))
    preTrainedParams.append((W, b))

try:
    # Maybe not the best way to do things...
    if forceCreate:
        raise IOError()

    postTrainedLayers = load_model('models/posttrained.h5', rng, batchSize, x, inputShape, nkerns, filterShape)
    models = build_lenet_model(datasets, postTrainedLayers, x, y, batchIndex, batchSize, learningRate)
except IOError:
    models = build_lenet_model(datasets, preTrainedLayers, x, y, batchIndex, batchSize, learningRate)
    trainModel = models[0]
    # Now train the algorithm using SGD
    postTrainedLayers = train_lenet5(datasets[0], trainModel, preTrainedLayers, rng, batchSize, batchIndex)
    save_model(postTrainedLayers, 'models/posttrained.h5')

postTrainedParams = []
for postTrainedLayer in postTrainedLayers:
    W = postTrainedLayer.W.get_value(borrow=False)
    b = postTrainedLayer.b.get_value(borrow=False)
    postTrainedParams.append((W, b))

# First check the accuracy of the model.
validateModel = models[1]

nValidateBatches = datasets[1][0].get_value(borrow=True).shape[0] // batchSize

# Run the validation set.
validationLosses = [
    validateModel(i)
    for i in range(nValidateBatches)
]

validationScore = np.mean(validationLosses)

print 'Validation score %.2f%%' % (validationScore * 100.)

# Next compare the anchor vectors of the pre-training and after-training layers.

# Print some whitespace to make things look nice.
for i in range(4):
    print ''


layers = zip(preTrainedParams, postTrainedParams)
# Only care about the first two conv/pooling layers
layers = layers[0:2]
for i, layer in enumerate(layers):
    # Zero is the index for the weight.
    preTrainedLayer = layer[0][0]
    postTrainedLayer = layer[1][0]

    print np.array(preTrainedLayer).shape

    print 'Angle differences for conv/pool layer %i' % (i)

    preTrainedFilters = [filterWeight.flatten() for filterWeight in preTrainedLayer]
    postTrainedFilters = [filterWeight.flatten() for filterWeight in postTrainedLayer]

    # Find the angle between each of the two filters.
    trainedFilters = zip(preTrainedFilters, postTrainedFilters)
    filterAngles = []
    for j, trainedFilter in enumerate(trainedFilters):
        angle = angle_between(trainedFilter[0], trainedFilter[1])
        filterAngles.append(angle)
        print '     Filter %i) %.4f' % (j, angle)

    avgAngle = np.mean(filterAngles)
    print '     Average angle) %.4f' % (avgAngle)
