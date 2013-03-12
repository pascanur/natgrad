import theano
import numpy
import time


class DataMNIST(object):
    def __init__(self, path, mbs, bs, rng, unlabled):
        self.path = path
        self.mbs = mbs
        self.bs = bs
        self.rng = rng
        self.unlabled = unlabled
        self.data = numpy.load(path)
        self.xdim = self.data['train_x'].shape[1]
        self.ydim = numpy.max(self.data['train_y'])+1

        self.offset = theano.shared(numpy.int32(0))
        self.begin = self.offset * self.mbs
        self.end = self.offset*self.mbs + self.mbs
        self._train_x = theano.shared(self.data['train_x'], name='train_x')
        self._train_y = theano.shared(self.data['train_y'], name='train_y')
        self._valid_x = theano.shared(self.data['valid_x'], name='valid_x')
        self._valid_y = theano.shared(self.data['valid_y'], name='valid_y')
        self._test_x = theano.shared(self.data['test_x'], name='test_x')
        self._test_y = theano.shared(self.data['test_y'], name='test_y')
        # Codes:
        # 0 -> same minibatch
        # 1 -> different minibatch
        # 2 -> validation set
        if unlabled == 0:
            self._natgrad = self._train_x[self.begin:self.end]
            self._natgrady = self._train_y[self.begin:self.end]
        elif unlabled == 1:
            self._natgrad = self._train_x[self.begin:self.end]
            self._natgrady = self._train_y[self.begin:self.end]
        elif unlabled == 2:
            self._natgrad = self._valid_x
            self._natgrady = self.valid_y

        self.eval_variables = [self._train_x,
                               self._train_y]
        self.n_valid_samples = self.data['valid_x'].shape[0]
        self.n_test_samples = self.data['test_x'].shape[0]

        self.n_batches = 50000 // self.bs
        self.nat_batches = self.n_batches

        if self.unlabled ==2:
            self.nat_batches = 10000 // self.mbs
        self.grad_perm = self.rng.permutation(self.n_batches)
        self.nat_perm = self.rng.permutation(self.nat_batches)
        self.variables = [self.train_x, self.train_y]
        self.train_variables = [
            self._train_x[self.offset*self.bs:
                          self.offset*self.bs+self.bs],
            self._train_y[self.offset*self.bs:
                          self.offset*self.bs+self.bs]]
        self.pos = -1
        self.nat_pos = -1

    def train_x(self, start, end):
        return self._train_x[
            self.offset*self.bs+start:self.offset*self.bs+end]

    def train_y(self, start, end):
        return self._train_y[
            self.offset*self.bs+start:self.offset*self.bs+end]

    def valid_x(self, start, end):
        return self._valid_x[start:end]

    def valid_y(self, start, end):
        return self._valid_y[start:end]

    def test_x(self, start, end):
        return self._test_x[start:end]

    def test_y(self, start, end):
        return self._test_y[start:end]

    def update_before_computing_gradients(self):
        self.pos = (self.pos + 1) % self.n_batches
        if self.pos % self.n_batches == 0:
            self.grad_perm = self.rng.permutation(self.n_batches)
        self.offset.set_value(self.grad_perm[self.pos])



    def update_before_computing_natural_gradients(self):
        if self.unlabled == 1:
            self.nat_pos = (self.nat_pos + 1) % self.nat_batches
            if self.nat_pos % self.nat_batches == 0:
                self.nat_perm = self.rng.permutation(self.nat_batches)
            self.offset.set_value(self.nat_perm[self.nat_pos])

    def update_before_evaluation(self):
        if self.unlabled == 1:
            self.offset.set_value(self.grad_perm[self.pos])
