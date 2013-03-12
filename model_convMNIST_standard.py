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
from conv_utils import LeNetConvPoolLayerStandard
from mlp_utils import HiddenLayerStandard, SoftmaxLayerStandard


class convMat(object):
    def __init__(self, state, data):
        self.rng = numpy.random.RandomState(state['seed'])
        self.srng = RandomStreams(self.rng.randint(1e5))
        self.data = data
        self.nin = data.xdim
        self.state = state
        self.nout = data.ydim
        nkerns = eval(str(state['nkerns']))

        #######################
        # 0. Training functions
        #######################
        self.x = TT.matrix('X')
        self.y = TT.ivector('y')
        layer0_input = self.x.reshape((state['cbs'], 1, 28, 28))
        self.layer0 = LeNetConvPoolLayerStandard(
            self.rng,
            input=layer0_input,
            image_shape=(state['cbs'], 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2),
            name='layer0')

        self.layer1 = LeNetConvPoolLayerStandard(
            self.rng,
            input=self.layer0.output,
            image_shape=(state['cbs'], nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2),
            name='layer1')

        layer2_input = self.layer1.output.flatten(2)
        self.layer2 = HiddenLayerStandard(
            self.rng,
            layer2_input,
            nkerns[1] * 4 * 4,
            eval(str(state['nhid'])))

        self.layer3 = SoftmaxLayerStandard(
            self.rng,
            self.layer2.output,
            eval(str(state['nhid'])),
            self.nout)

        self.params = []
        self.params += self.layer0.params
        self.params += self.layer1.params
        self.params += self.layer2.params
        self.params += self.layer3.params
        self.params_shape = [x.get_value(borrow=True).shape
                             for x in self.params]
        ##### PARAMS
        self.inputs = [self.x, self.y]
        inds = TT.constant(numpy.asarray(range(state['cbs']),
                                         dtype='int32'))
        cost = -TT.log(self.layer3.output)[inds, self.y]
        self.train_cost = TT.mean(cost)
        self.Gvs = lambda *args:\
                TT.Lop(self.layer3.output,
                       self.params,
                       TT.Rop(self.layer3.output,
                              self.params,
                              args) / (self.layer3.output * state['mbs']))

        pred = TT.argmax(self.layer3.output, axis=1)
        self.error = TT.mean(TT.neq(pred, self.y)) * 100.

        #########################
        # 1. Validation functions
        #########################
        self.eval_x = TT.matrix('X')
        self.eval_y = TT.ivector('y')
        layer0_input = self.eval_x.reshape((1, 1, 28, 28))
        self.eval_layer0 = LeNetConvPoolLayerStandard(
            self.rng,
            input=layer0_input,
            image_shape=(1, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2),
            name='layer0',
            W=self.layer0.W,
            b=self.layer0.b)

        self.eval_layer1 = LeNetConvPoolLayerStandard(
            self.rng,
            input=self.eval_layer0.output,
            image_shape=(1, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2),
            name='layer1',
            W=self.layer1.W,
            b=self.layer1.b)

        layer2_input = self.eval_layer1.output.flatten(2)
        self.eval_layer2 = HiddenLayerStandard(
            self.rng,
            layer2_input,
            nkerns[1] * 4 * 4,
            eval(str(state['nhid'])),
            W=self.layer2.W,
            b=self.layer2.b)

        self.eval_layer3 = SoftmaxLayerStandard(
            self.rng,
            self.eval_layer2.output,
            eval(str(state['nhid'])),
            self.nout,
            W=self.layer3.W,
            b=self.layer3.b)
        pred = TT.argmax(self.eval_layer3.output, axis=1)
        err = TT.mean(TT.neq(pred, self.eval_y)) * 100
        givens = {}
        entry = TT.iscalar('entry')
        givens[self.eval_x] = self.data.valid_x(entry, entry + 1)
        givens[self.eval_y] = self.data.valid_y(entry, entry + 1)
        self.valid_eval_func = theano.function([entry],
                                               err,
                                               givens=givens,
                                               name='valid_eval_fn',
                                               profile=0)

        givens[self.eval_x] = self.data.test_x(entry, entry + 1)
        givens[self.eval_y] = self.data.test_y(entry, entry + 1)
        self.test_eval_func = theano.function([entry],
                                    err,
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
