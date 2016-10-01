import numpy
import theano
import theano.tensor as T


class HiddenLayer(object):
    def __init__(self, rng, input, nIn, nOut, activation=T.tanh, setParams=None):
        self.input = input
        if not setParams is None:
            W, b = setParams
        else:
            W = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (nIn + nOut)),
                    high=numpy.sqrt(6. / (nIn + nOut)),
                    size=(nIn, nOut)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                wValues *= 4
            b = numpy.zeros((nOut,), dtype=theano.config.floatX)

        self.W = theano.shared(value=W, name='W', borrow=True)
        self.b = theano.shared(value=b, name='b', borrow=True)


        linOutput = T.dot(self.input, self.W) + self.b
        self.output = (
            linOutput if activation is None
            else activation(linOutput)
        )

        self.params = [self.W, self.b]
