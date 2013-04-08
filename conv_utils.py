import numpy
import theano.tensor as TT
import theano
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import time


class FastConvLayer(object):
    def __init__(self,
                 rng,
                 input,
                 irange,
                 input_channels,
                 output_channels,
                 kernel_shape,
                 input_axes = ('c',0,1,'b'),
                 output_axes = ('c', 0,1,'b'),
                 subsample = (1,1),
                 pad = 0,
                 partial_sum = None,
                 name = 'tmp',
                 activ = TT.nnet.sigmoid,
                 image_shape = None,
                 pool_stride = None,
                 pool_shape = None,
                 tied_b = False,
                 filters = None,
                 b = None):

        from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b
        from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
        from theano.sandbox.cuda.basic_ops import gpu_contiguous
        from theano.sandbox.cuda import gpu_from_host
        from theano.sandbox.cuda import host_from_gpu
        output_shape = [image_shape[0] + 2 * pad - kernel_shape[0] + 1,
                        image_shape[1] + 2 * pad - kernel_shape[1] + 1]

        self.image_shape = image_shape
        self.tied_b = tied_b
        if filters is None:
            filters = theano.shared(numpy.float32(
                rng.uniform(-irange, irange, (input_channels, \
                                              kernel_shape[0],
                                              kernel_shape[1],
                                              output_channels))),
            name='W%s'%name)

        if b is None:
            if tied_b:
                b = theano.shared(numpy.zeros((output_channels,),
                                            dtype='float32'),
                                  name='b%s'%name)


            else:
                #xval = int(numpy.ceil((float(output_shape[0]) -
                #                       pool_shape[0])/pool_stride[0])+1)
                #yval = int(numpy.ceil((float(output_shape[1]) -
                #                       pool_shape[1])/pool_stride[1])+1)
                #print '-------------------------'
                #print xval, yval
                #print
                b = theano.shared(numpy.zeros((output_channels,
                                               output_shape[0],
                                               output_shape[1]),
                                             dtype='float32'),
                                  name='b%s'%name)
        self.b =b
        if tied_b:
            nb = b.dimshuffle(0,'x','x','x')

        else:
            nb = b.dimshuffle(0,1,2,'x')
        if subsample != (1, 1):

            raise NotImplementedError()

        self.input_axes = input_axes
        self.output_axes = output_axes
        if 'Cuda' not in str(type(filters)):
            self.W = gpu_from_host(filters)
        else:
            self.W = filters
        self.pad = pad
        self.partial_sum = partial_sum
        cpu = False
        if 'Cuda' not in str(type(input)):
            cpu = True
            input = gpu_from_host(input)

        assert input.ndim == 4
        x_axes = self.input_axes
        op_axes = ('c',0,1,'b')
        if tuple(x_axes) != op_axes:
            input = input.dimshuffle(*[x_axes.index(axis) for axis in
                                       op_axes])
        input = gpu_contiguous(input)
        rval = FilterActs(self.pad, self.partial_sum)(input, self.W)

        # Format the output based on the output space
        rval_axes = self.output_axes
        assert len(rval_axes) == 4

        if tuple(rval_axes) != op_axes:
            rval = rval.dimshuffle(*[op_axes.index(axis) for axis in rval_axes])

        if cpu:
            rval = host_from_gpu(rval)


        rval = max_pool_c01b(c01b = rval + nb,
                          pool_shape = pool_shape,
                          pool_stride = pool_stride,
                          image_shape = image_shape)
        rval = activ(rval)

        self.output = rval
        self.params = [self.W, self.b]

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
