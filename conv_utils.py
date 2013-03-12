import numpy
import theano.tensor as TT
import theano
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import time


class LeNetConvPoolLayerStandard(object):

    def __init__(self,
                 rng,
                 input,
                 filter_shape,
                 image_shape,
                 poolsize=(2, 2),
                 name='conv',
                 W=None,
                 b=None):

        assert image_shape[1] == filter_shape[1]
        self.input = input
        if W is None:
            W_values = numpy.zeros(filter_shape, dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name=name + '_W')
        else:
            self.W = W
        if b is None:
            b_values = numpy.zeros((filter_shape[0],),
                                   dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name=name + '_b')

        else:
            self.b = b
        conv_out = conv.conv2d(input=input,
                               filters=self.W,
                               filter_shape=filter_shape,
                               image_shape=image_shape)

        if W is None:
            fan_in = numpy.prod(filter_shape[1:])
            fan_out = filter_shape[0] * numpy.prod(filter_shape[2:]) /\
                    numpy.prod(poolsize)
            W_bound = numpy.sqrt(6. / (fan_in + fan_out)) * 4.
            values = numpy.asarray(rng.uniform(low=-W_bound,
                                               high=W_bound,
                                               size=filter_shape),
                                   dtype=theano.config.floatX)
            self.W.set_value(values, borrow=True)

        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize,
                                            ignore_border=True)
        self.output = TT.nnet.sigmoid(pooled_out +
                                      self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]


class LeNetConvPoolLayerLW(object):

    def __init__(self,
                 rng,
                 input,
                 filter_shape,
                 image_shape,
                 poolsize=(2, 2),
                 name='conv',
                 W=None,
                 b=None):

        assert image_shape[1] == filter_shape[1]
        self.input = input
        if W is None:
            W_values = numpy.zeros(filter_shape, dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, name=name + '_W')
        else:
            self.W = W
        if b is None:
            b_values = -2 * numpy.ones((filter_shape[0],),
                                     dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name=name + '_b')

        else:
            self.b = b
        conv_out = conv.conv2d(input=input,
                               filters=self.W,
                               filter_shape=filter_shape,
                               image_shape=image_shape)

        if W is None:
            fan_in = numpy.prod(filter_shape[1:])
            fan_out = filter_shape[0] * numpy.prod(filter_shape[2:]) /\
                    numpy.prod(poolsize)
            W_bound = numpy.sqrt(6. / (fan_in + fan_out)) * 4.
            values = numpy.asarray(rng.uniform(low=-W_bound,
                                               high=W_bound,
                                               size=filter_shape),
                                   dtype=theano.config.floatX)
            for dx in xrange(filter_shape[0]):
                for kx in xrange(filter_shape[1]):
                    s, v, d = numpy.linalg.svd(values[dx][kx],
                                   full_matrices=False)
                    values[dx][kx] = numpy.dot(s * 4, v)
            self.W.set_value(values, borrow=True)

        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize,
                                            ignore_border=True)
        self.output = TT.nnet.sigmoid(pooled_out +
                                      self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
