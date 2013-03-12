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


class mlp(object):
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
        self.x = TT.matrix('X')
        self.y = TT.ivector('y')
        self.layer0 = HiddenLayerStandard(
            self.rng,
            self.x,
            self.nin,
            eval(str(state['nhid'])),
            name='layer0')

        self.layer1 = SoftmaxLayerStandard(
            self.rng,
            self.layer0.output,
            eval(str(state['nhid'])),
            self.nout)

        self.params = []
        self.params += self.layer0.params
        self.params += self.layer1.params
        self.best_params = [(x.name, x.get_value()) for x in self.params]
        self.params_shape = [x.get_value(borrow=True).shape
                             for x in self.params]
        ##### PARAMS
        self.inputs = [self.x, self.y]
        inds = TT.constant(numpy.asarray(range(state['cbs']),
                                         dtype='int32'))
        cost = -TT.log(self.layer1.output)[inds, self.y]
        self.train_cost = TT.mean(cost)
        if state['matrix'] == 'KL':
            self.Gvs = lambda *args:\
                    TT.Lop(self.layer1.output,
                           self.params,
                           TT.Rop(self.layer1.output,
                                  self.params,
                                  args) / (self.layer1.output * state['mbs']))
        elif state['matrix'] == 'cov':
            self.Gvs = lambda *args:\
                    TT.Lop(cost,
                           self.params,
                           TT.Rop(cost,
                                  self.params,
                                  args) / (numpy.float32(state['mbs'])))

        pred = TT.argmax(self.layer1.output, axis=1)
        self.error = TT.mean(TT.neq(pred, self.y)) * 100.

        #########################
        # 1. Validation functions
        #########################
        givens = {}
        givens[self.x] = self.data._valid_x
        givens[self.y] = self.data._valid_y
        self.valid_eval_func = theano.function([],
                                               self.error,
                                               givens=givens,
                                               name='valid_eval_fn',
                                               profile=0)

        givens[self.x] = self.data._test_x
        givens[self.y] = self.data._test_y
        self.test_eval_func = theano.function([],
                                    self.error,
                                    givens=givens,
                                    name='test_fn',
                                    profile=0)

    def validate(self):
        return self.valid_eval_func()

    def  test_eval(self):
        return self.test_eval_func()

    def save(self, filename):
        vals = dict([(x.name, x.get_value()) for x in self.params])
        for name, val in self.best_params:
            vals['b'+name] = val
        numpy.savez(filename, **vals)

    def load(self, filename):
        values = numpy.load(filename)['arr_0'].item()
        for param in self.params:
            param.set_value(values[param.name], borrow=True)
