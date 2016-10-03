from experiment import *
from data_helper import load_data
import numpy as np
from model_builder import build_lenet_layers
from model_builder import build_lenet_model

filterShape = (5,5)
stride = (1,1)
inputShape = (28,28)
nkerns = (6, 16)
rng = np.random.RandomState()
batchSize = 2
nUseSamples = 100
learningRate = 0.1

batchIndex = T.lscalar()
x = T.matrix('x')
y = T.ivector('y')

datasets = load_data(nUseSamples, 'mnist.pkl')

try:
    preTrainedLayers = load_model('models/pretrained.h5', rng, batchSize, x, inputShape, nkerns, filterShape)
except IOError:
    preTrainedLayers = create_pretrained(datasets, filterShape, stride, inputShape, nkerns, rng, batchSize, x, y, batchIndex)
    save_model(preTrainedLayers, 'models/pretrained.h5')

try:
    postTrainedLayers = load_model('models/posttrained.h5', rng, batchSize, x, inputShape, nkerns, filterShape)
    models = build_lenet_model(datasets, postTrainedLayers, x, y, batchIndex, batchSize)
except IOError:
    models = build_lenet_model(datasets, preTrainedLayers, x, y, batchIndex, batchSize, learningRate)
    trainModel = models[0]
    # Now train the algorithm using SGD
    postTrainedLayers = train_lenet5(datasets[0], trainModel, preTrainedLayers, rng, batchSize, batchIndex)
    save_model(postTrainedLayers, 'models/posttrained.h5')

# First check the accuracy of the model.
validateModel = models[1]

nValidateBatches = datasets[1][0].get_value(borrow=True).shape[0] // batchSize

# Run the validation set.
validationLosses = [
    validateModel(i)
    for i in range(nValidateBatches)
]

validationScore = numpy.mean(validationLosses)

print 'Validation score %.2f%%' % (validationScore * 100.)
