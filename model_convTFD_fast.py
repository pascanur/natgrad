"""
Razvan Pascanu
"""
import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle
import gzip
import numpy

from utils import softmax
from conv_utils import LeNetConvPoolLayerStandard, FastConvLayer
from mlp_utils import HiddenLayerStandard, HiddenLayerLW, SoftmaxLayerStandard


class convMat(object):
    def __init__(self, state, data):
        self.rng = numpy.random.RandomState(state['seed'])
        self.srng = RandomStreams(self.rng.randint(1e5))
        self.data = data
        self.nin = data.xdim
        self.state = state
        self.nout = data.ydim

        #######################
        # 0. Training functions
        #######################
        #self.x = TT.tensor4('X')
        self.x = TT.matrix('X')
        self.y = TT.ivector('y')
        layer0_input = self.x
        layer0_input = self.x.reshape(
            (state['cbs'], 1, 48, 48)).dimshuffle((1,2,3,0))
        self.layer0 = FastConvLayer(
            self.rng,
            input=layer0_input,
            irange=.2,
            pad = 0,
            activ = lambda x: TT.switch(x>0, x, 0),
            input_channels=1,
            output_channels=16,
            kernel_shape=[9,9],
            image_shape = [48,48],
            name = 'l0',
            #partial_sum = 4,
            tied_b=False,
            pool_stride = [5,5],
            pool_shape = [5,5])
        # 48  - 9 + 1 = 40
        # (40 - 5)/5 + 1 = 8
        layer1_input = self.layer0.output.dimshuffle(3,1,2,0).flatten(2)
        self.layer1 = HiddenLayerLW(
            self.rng,
            layer1_input,
            16 * 8 * 8,
            256,
            name = 'l1',
            activation = TT.tanh
            )

        self.layer2 = HiddenLayerLW(
            self.rng,
            self.layer1.output,
            256,
            self.nout,
            name='l2',
            activation = TT.nnet.sigmoid
            )

        self.params = []
        self.params += self.layer0.params
        self.params += self.layer1.params
        self.params += self.layer2.params
        self.params_shape = [x.get_value(borrow=True).shape
                             for x in self.params]
        ##### PARAMS
        self.inputs = [self.x, self.y]
        membuf = TT.shared(numpy.zeros((state['cbs'], self.nout),
                                       dtype=theano.config.floatX),
                           name='membuf')
        expanded_y = TT.set_subtensor(
            membuf[numpy.asarray(range(state['cbs']), dtype='int32'),
                   self.y], numpy.ones((state['cbs'],),
                                       dtype=theano.config.floatX))

        cost = - TT.log(self.layer2.output)*expanded_y - \
                TT.log(1-self.layer2.output)*(1-expanded_y)

        self.train_cost = TT.mean(cost)
        self.Gvs = lambda *args:\
                TT.Lop(self.layer2.output,
                       self.params,
                       TT.Rop(self.layer2.output,
                              self.params,
                              args) / ((1-self.layer2.output)*self.layer2.output * state['mbs']))

        pred = TT.argmax(self.layer2.output, axis=1)
        self.error = TT.mean(TT.neq(pred, self.y)) * 100.

        #########################
        # 1. Validation functions
        #########################

        inds = TT.constant(numpy.asarray(range(1),
                                         dtype='int32'))
        pred = TT.argmax(self.layer2.output, axis=1)
        error = theano.clone(
            TT.mean(TT.neq(pred, self.y)) * 100.,
            replace={layer0_input:
                     TT.unbroadcast(self.x.reshape(
                (1,1,48,48)),0).dimshuffle((1,2,3,0))})

        self.cbs = state['cbs']
        givens = {}
        entry = TT.iscalar('entry')
        givens[self.x] = self.data.valid_x(entry, (entry + 1))
        givens[self.y] = self.data.valid_y(entry, (entry + 1))
        self.valid_eval_func = theano.function([entry],
                                               error,
                                               givens=givens,
                                               name='valid_eval_fn',
                                               profile=0)

        givens[self.x] = self.data.test_x(entry,
                                               (entry + 1))
        givens[self.y] = self.data.test_y(entry,
                                               (entry + 1))
        self.test_eval_func = theano.function([entry],
                                    error,
                                    givens=givens,
                                    name='test_fn',
                                    profile=0)


    def validate(self):

        return numpy.mean([self.valid_eval_func(k) for k in
                          xrange(self.data.n_valid_samples)])

    def  test_eval(self):
        return numpy.mean([self.test_eval_func(k) for k in
                          xrange(self.data.n_test_samples)])

    def save(self, filename):
        vals = dict([(x.name, x.get_value()) for x in self.params])
        numpy.savez(filename, vals)

    def load(self, filename):
        values = numpy.load(filename)
        for param in self.params:
            param.set_value(values[param.name], borrow=True)
