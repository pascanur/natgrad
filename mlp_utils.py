import numpy
import theano.tensor as TT
import theano
import time

from utils import softmax


class HiddenLayerStandard(object):
    def __init__(self, rng, input, n_in, n_out, activation=TT.nnet.sigmoid,
                 W=None, b=None, name='tmp'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                              layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        if W is None and b is None:
            if activation == theano.tensor.tanh:
                W_values = numpy.asarray(rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)), dtype=theano.config.floatX)
            elif activation == theano.tensor.nnet.sigmoid:
                W_values = numpy.asarray(4 * rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)), dtype=theano.config.floatX)
            else:
                W_values = numpy.asarray(rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)), dtype=theano.config.floatX)

            self.W = theano.shared(value=W_values, name=name+'_W')

            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name=name+'_b')
        else:
            self.W = W
            self.b = b

        self.output = activation(TT.dot(input, self.W) + self.b)
        # parameters of the model
        self.params = [self.W, self.b]


class HiddenLayerLW(object):
    def __init__(self, rng, input, n_in, n_out, activation=TT.tanh,
                 W=None, b=None, name='tmp'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                              layer
        """
        self.input = input
        if W is None and b is None:
            if activation == theano.tensor.tanh:
                W_values = numpy.asarray(rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)), dtype=theano.config.floatX)
                s, v, d = numpy.linalg.svd(W_values, full_matrices=False)
                W_values = numpy.dot(s, d)
                b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            elif activation == theano.tensor.nnet.sigmoid:
                W_values = numpy.asarray(4 * rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)), dtype=theano.config.floatX)
                s, v, d = numpy.linalg.svd(W_values, full_matrices=False)
                W_values = numpy.dot(s * 4, d)
                b_values = -2 * numpy.ones((n_out,), dtype=theano.config.floatX)
            else:
                W_values = numpy.asarray(rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)), dtype=theano.config.floatX)

            self.W = theano.shared(value=W_values, name=name+'_W')
            self.b = theano.shared(value=b_values, name=name+'_b')
        else:
            self.W = W
            self.b = b
        self.output = activation(TT.dot(input, self.W) + self.b)
        # parameters of the model
        self.params = [self.W, self.b]


class SoftmaxLayerStandard(object):
    def __init__(self, rng, input, n_in, n_out,
                 W=None, b=None, name='tmp'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units
        """
        self.input = input
        if W is None and b is None:
            W_values = numpy.asarray(rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)), dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name=name+'_W')
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name=name+'_b')
        else:
            self.W = W
            self.b = b

        self.output = softmax(TT.dot(input, self.W) + self.b)
        # parameters of the model
        self.params = [self.W, self.b]

