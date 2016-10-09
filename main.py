# API imports
import copy
import numpy as np
from theano import tensor as T

# User defined imports
from data_helper import load_data
from trained_model import TrainedModel
from pretrained_model import PretrainedModel
from experiment_model import get_anchor_angles

filterShape = (5,5)
stride = (1,1)
inputShape = (1,28,28)
nkerns = (6, 16)
rng = np.random.RandomState()
batchSize = 500
nUseSamples = 10000
learningRate = 0.1
nEpochs = 200
forceCreate = True

# For the pretrained model.
batchIndex0 = T.lscalar()
x0 = T.matrix('x')
y0 = T.ivector('y')

datasets = load_data(nUseSamples, 'mnist.pkl')
nValidateBatches = datasets[1][0].get_value(borrow=True).shape[0] // batchSize
print nValidateBatches

preTrainedModel = PretrainedModel(forceCreate, rng, batchSize, batchIndex0, x0, y0,
                                nkerns, filterShape, stride, inputShape, datasets, learningRate)

validationScore = preTrainedModel.get_validation_score(nValidateBatches)

print 'Validation score %.2f%%' % (validationScore * 100.)
