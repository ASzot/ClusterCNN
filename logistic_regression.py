import numpy
import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, input, nIn, nOut, setParams=None):
        if not setParams is None:
            W, b = setParams
        else:
            W = numpy.zeros(
                (nIn, nOut),
                dtype = theano.config.floatX
            )
            b = numpy.zeros(
                (nOut, ),
                dtype = theano.config.floatX
            )

        self.W = theano.shared(
            value=W,
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value = b,
            name='b',
            borrow=True
        )

        self.input = input

        self.pOfYGivenX = T.nnet.softmax(T.dot(self.input, self.W) + self.b)

        self.yPred = T.argmax(self.pOfYGivenX, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.pOfYGivenX)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.yPred.ndim:
            raise TypeError('y should have the same shape as self.yPred')
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.yPred, y))
        else:
            raise NotImplementedError()
