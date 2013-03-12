import numpy
import time

import theano
import theano.tensor as TT
from theano.gof import local_optimizer
from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from minres import minres, minres_messages
from minres import minresQLP, minresQLP_messages
from utils import forloop, safe_clone, print_time, print_mem, const


class SGD(object):
    def __init__(self,
                 model,
                 state,
                 data):
        """
        Parameters:
            :param model:
                Class describing the model used.  It should provide the
                 computational graph to evaluate the model
            :param state:
                Dictionary containing the current state of your job. This
                includes configuration of the job, specifically the seed,
                the startign damping factor, batch size, etc. See main.py
                for details
            :param data:
                Class describing the dataset used by the model
        """

        #####################################
        # Step 0. Constructs shared variables
        #####################################
        n_params = len(model.params)
        cbs = state['cbs']
        bs = state['bs']
        mbs = state['mbs']
        ebs = state['ebs']
        profile = state['profile']
        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])
        srng = RandomStreams(self.rng.randint(213))

        self.gs = [theano.shared(numpy.zeros(shp, dtype=theano.config.floatX))
                   for shp in model.params_shape]

        self.loop_inps = [theano.shared(
            numpy.zeros(shp, dtype=theano.config.floatX))
            for shp in model.params_shape]
        self.loop_outs = [theano.shared(
            numpy.zeros(shp, dtype=theano.config.floatX))
            for shp in model.params_shape]
        self.step = 0
        self.cbs = cbs
        self.bs = bs
        self.mbs = mbs
        self.ebs = ebs
        self.state = state
        self.profile = profile
        self.data = data
        self.step_timer = time.time()

        ############################################################
        # Step 1. Compile function for computing eucledian gradients
        ############################################################
        print 'Constructing grad function'
        bdx = TT.iscalar('batch_idx')
        loc_data = [x(bdx * cbs, (bdx + 1) * cbs) for x in
                    self.data.variables]
        cost = safe_clone(model.train_cost, model.inputs, loc_data)
        gs = TT.grad(cost, model.params)
        ratio = numpy.float32(float(bs) / cbs)
        update = [(g, g + lg / ratio) for g, lg in zip(self.gs, gs)]
        print 'Compiling grad function'
        st = time.time()
        self.loc_grad_fn = theano.function(
            [bdx], [], updates=update, name='loc_fn_grad', profile=profile)
        print 'took', time.time() - st

        norm_grads = TT.sqrt(sum(TT.sum(x ** 2) for x in self.gs))
        ###########################################################
        # Step 3. Compile function for evaluating cost and updating
        # parameters
        ###########################################################
        print 'constructing evaluation function'
        lr = TT.scalar('lr')
        self.lr = numpy.float32(state['lr'])
        loc_data = [x(bdx * cbs, (bdx + 1) * cbs) for x in
                    self.data.variables]
        old_cost = safe_clone(model.train_cost, model.inputs, loc_data)
        self.loc_old_cost = theano.function(
            [bdx], old_cost, name='loc_old_cost', profile=profile)
        new_params = [p - lr * r for p, r in zip(model.params, self.gs)]
        new_cost = safe_clone(model.train_cost,
                              model.inputs + model.params,
                              loc_data + new_params)
        new_err = safe_clone(model.error,
                             model.inputs + model.params,
                             loc_data + new_params)
        self.loc_new_cost = theano.function(
            [bdx, lr], [new_cost, new_err], name='loc_new_cost',
            profile=profile)

        loc_data = [x[bdx * cbs: (bdx + 1) * cbs] for x in
                    self.data.eval_variables]
        new_cost = safe_clone(model.train_cost,
                              model.inputs + model.params,
                              loc_data + new_params)
        new_err = safe_clone(model.error,
                             model.inputs + model.params,
                             loc_data + new_params)
        self.loc_new_cost_all = theano.function(
            [bdx, lr], [new_cost, new_err], name='loc_new_cost',
            profile=profile)
        self.update_params = theano.function(
            [lr], [], updates=zip(model.params, new_params),
            name='update_params')
        old_cost = TT.scalar('old_cost')
        new_cost = TT.scalar('new_cost')
        dist = -lr * sum([TT.sum(g * r) for g, r in zip(self.gs, self.gs)])
        rho = (new_cost - old_cost) / dist
        self.compute_rho = theano.function(
            [old_cost, new_cost, lr], [rho, norm_grads], name='compute_rho', profile=profile)
        self.old_cost = 1e20
        self.return_names = ['cost',
                             'error',
                             'time_grads',
                             'time_eval',
                             'norm_grad',
                             'rho',
                             'lr']

    def compute_gradients(self):
        for g in self.gs:
            g.container.storage[0][:] = 0
        for idx in xrange(self.bs // self.cbs):
            self.loc_grad_fn(idx)

    def compute_old_cost(self):
        costs = [self.loc_old_cost(idx)
                 for idx in xrange(self.bs // self.cbs)]
        return numpy.mean(costs).astype(theano.config.floatX)

    def compute_new_cost(self, lr):
        rvals = [self.loc_new_cost(idx, self.lr)
                 for idx in xrange(self.bs // self.cbs)]
        cost = numpy.mean([x for x, y in
                            rvals]).astype(theano.config.floatX)
        error = numpy.mean([y for x, y in
                             rvals]).astype(theano.config.floatX)
        return cost, error

    def compute_new_cost_all(self, lr):
        rvals = [self.loc_new_cost_all(idx, self.lr)
                 for idx in xrange(self.ebs // self.cbs)]
        cost = numpy.mean([x for x, y in
                            rvals]).astype(theano.config.floatX)
        error = numpy.mean([y for x, y in
                             rvals]).astype(theano.config.floatX)
        self._ocost = 0
        self._ncost = 0
        self._error = 0
        self._etime = 0
        return cost, error

    def __call__(self):
        self.data.update_before_computing_gradients()
        g_st = time.time()
        self.compute_gradients()
        g_ed = time.time()
        self.data.update_before_computing_natural_gradients()
        if self.state['lr_adapt'] == 1:
            if self.step > self.state['lr_adapt_start']:
                if self.step % self.state['lr_adapt_change'] == 0:
                    self.lr = self.lr * self.state['lr_adapt_decrease']
        elif self.state['lr_adapt'] == 2:
            if self.step > self.state['lr_adapt_start']:
                self.lr = self.state['lr0'] /\
                    (1. + float(self.step - self.state['lr_adapt_start'])/self.state['lr_beta'])
                self.state['lr'] = float(self.lr)
        if self.step % self.state['trainFreq'] == 0:
            e_st = time.time()
            old_cost = self.compute_old_cost()
            new_cost, error = self.compute_new_cost(self.lr)
            rho, norm_grad = self.compute_rho(old_cost, new_cost, self.lr)
            new_cost, error = self.compute_new_cost_all(self.lr)

            if new_cost > self.state['btraincost'] * 6:
                raise Exception('Variance too large on training cost!')

            while (numpy.isnan(new_cost) or
                   numpy.isinf(new_cost)):
                raise Exception('Learning rate too small !')
            self.old_cost = new_cost
            self.update_params(self.lr)
            e_ed = time.time()
            self._ocost = old_cost
            self._ncost = new_cost
            self._error = error
            self._eetime = e_ed
            self._estime = e_st
            msg = ('.. iter %4d cost %.3g, error %.3g step_size %.3g '
                   'rho %.3g '
                   'norm grad %.3g '
                   'time [grad] %s,'
                   '[updates param] %s,'
                   'whole time %s'
                  )
            print msg % (
                self.step,
                new_cost,
                error,
                self.lr,
                rho,
                norm_grad,
                print_time(g_ed - g_st),
                print_time(e_ed - e_st),
                print_time(time.time() - self.step_timer) )
            self.step_timer = time.time()

        else:
            old_cost = self._ocost
            new_cost = self._ncost
            error = self._error
            e_ed = self._eetime
            e_st = self._estime
            rho, norm_grad = self.compute_rho(old_cost, new_cost, self.lr)
            self.update_params(self.lr)
        self.step += 1

        ret = {
            'cost': float(new_cost),
            'error': float(error),
            'time_grads': float(g_ed - g_st),
            'time_eval': float(e_ed - e_st),
            'norm_grad':norm_grad,
            'lr': self.lr,
            'rho': numpy.float32(rho)}
        return ret
