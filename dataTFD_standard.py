"""
Toronto Faces Dataset - v1.1
"""
import theano
import theano.tensor as TT
import numpy
import scipy.io
import time


class DataTFD_48(object):
    def __init__(self, path, mbs, bs, rng, fold=0, unlabled=0, normalize=1):
        self.path = path
        self.mbs = mbs
        self.bs = bs
        self.rng = rng
        self.unlabled = unlabled
        self.data = scipy.io.loadmat(path + 'TFD_48x48.mat')
        self.train_inds = self.data['folds'][:,fold] == 1
        self.valid_inds = self.data['folds'][:,fold] == 2
        self.test_inds = self.data['folds'][:, fold] == 3
        self.unlabeled_inds = self.data['folds'][:, fold] == 0

        self.xdim = 48*48
        self.offset = theano.shared(numpy.int32(0), 'offset')
        self.m_offset = theano.shared(numpy.int32(0), 'offset')

        self.ydim = numpy.max(self.data['labs_ex'])+1
        data_x = self.data['images'][self.train_inds]
        data_x = data_x.reshape((-1, 48*48))
        total_elems = data_x.shape[0]
        self.n_batches = data_x.shape[0] // bs
        self.n_offset = data_x.shape[0] - bs * self.n_batches + 1

        self.data_x = data_x.astype('float32')/numpy.float32(255)
        glob_min = 0
        glob_max = 1
        if normalize:
            self.data_x = self.data_x - self.data_x.mean(1).reshape(-1,1)
            self.data_x = self.data_x /\
                    numpy.float32(self.data_x.std(1).reshape(-1,1))
            glob_min = self.data_x.min()
            self.data_x = self.data_x - glob_min
            glob_max = self.data_x.max()
            self.data_x = self.data_x / glob_max
        data_y = self.data['labs_ex'][self.train_inds][:,0]
        data_y = data_y - numpy.min(data_y)
        self.data_y = data_y.astype('int32')

        self._train_x = theano.shared(self.data_x, 'eval_x')
        self._train_y = theano.shared(self.data_y, 'eval_y')
        self.eval_variables = [self._train_x, self._train_y]

        self._natgrad = self._train_x[self.offset:self.offset+self.bs]
        self._natgrady = self._train_y[self.offset:self.offset+self.bs]
        if unlabled > 1:
            self.unlabled_data = self.data['images'][self.unlabeled_inds].reshape((-1,48*48))
            self.unlabled_data = self.unlabled_data.astype('float32')/numpy.float32(255)
            if normalize:
                self.unlabled_data = self.unlabled_data -\
                    self.unlabled_data.mean(1).reshape(-1,1)
                self.unlabled_data = self.unlabled_data /\
                        numpy.float32(self.unlabled_data.std(1).reshape(-1,1)+1e-5)
                self.unlabled_data = self.unlabled_data - glob_min
                self.unlabled_data = self.unlabled_data / glob_max
            self.nat_values = self.unlabled_data[:mbs]

            self._natgrad = theano.shared(self.nat_values, borrow=True)
        if unlabled == 1:
            self.nat_batches = data_x.shape[0] // mbs
            self.nat_offset = data_x.shape[0] - mbs*self.nat_batches +1
            self.nat_values = self.data_x[:mbs]
            self.nat_valuesy = self.data_y[:mbs]
            self._natgrad = self._train_x[
                self.m_offset:self.m_offset+self.mbs]
            self._natgrady = self._train_y[
                self.m_offset:self.m_offset+self.mbs]
            self._natgrady = theano.shared(self.nat_valuesy, borrow=True)
            self._natgrad = theano.shared(self.nat_values, borrow=True)
        elif unlabled == 2:
            self.pv_perm = self.rng.permutation(self.n_batches)
            self.nat_batches = self.unlabled_data.shape[0] // (mbs - bs)
            self.nat_offset = 1
            self.nat_perm = self.rng.permutation(self.n_batches)
            self.nat_off = self.rng.randint(self.nat_offset)

        self.grad_perm = self.rng.permutation(self.n_batches)
        self.grad_off = self.rng.randint(self.n_offset)


        valid_x = self.data['images'][self.valid_inds]
        valid_x = valid_x.reshape((-1,48*48)).astype('float32')
        valid_x = valid_x / numpy.float32(255.)
        if normalize:
            valid_x = valid_x - valid_x.mean(1).reshape(-1,1)
            valid_x = valid_x / valid_x.std(1).reshape(-1,1)
            valid_x = valid_x - glob_min
            valid_x = valid_x / glob_max
        self._valid_x = theano.shared(
            valid_x, name='valid_x')
        valid_y = self.data['labs_ex'][self.valid_inds][:,0]
        valid_y = valid_y - numpy.min(valid_y)
        self._valid_y = theano.shared(valid_y.astype('int32'), name='valid_y')


        test_x = self.data['images'][self.test_inds]
        test_x = test_x.reshape((-1,48*48)).astype('float32')
        test_x = test_x / numpy.float32(255.)
        if normalize:
            test_x = test_x - test_x.mean(1).reshape(-1,1)
            test_x = test_x / test_x.std(1).reshape(-1,1)
            test_x = test_x - glob_min
            test_x = test_x / glob_max

        self._test_x = theano.shared(test_x, name='test_x')
        test_y = self.data['labs_ex'][self.test_inds][:,0]
        test_y = test_y - numpy.min(test_y)
        self._test_y = theano.shared(test_y.astype('int32'), name='test_y')
        self.n_valid_samples = valid_x.shape[0]
        self.n_test_samples = test_x.shape[0]
        print 'n_valid', self.n_valid_samples
        print 'n_test', self.n_test_samples
        self.variables = [self.train_x, self.train_y]
        self.posg = -1
        self.posng = -1

    def train_x(self, start, end):
        return self._train_x[start + self.offset:end + self.offset]

    def train_y(self, start, end):
        return self._train_y[start + self.offset:end + self.offset]

    def valid_x(self, start, end):
        return self._valid_x[start:end]

    def valid_y(self, start, end):
        return self._valid_y[start:end]

    def test_x(self, start, end):
        return self._test_x[start:end]

    def test_y(self, start, end):
        return self._test_y[start:end]

    def nat_train_x(self, start, end):
        if self.unlabled == 0:
            return self._train_x[start + self.offset:end + self.offset]
        else:
            return self.natgrad[start:end]

    def update_before_computing_gradients(self):
        self.posg = (self.posg+1) % self.n_batches
        if self.posg == 0:
            self.grad_perm = self.rng.permutation(self.n_batches)
            self.pv_perm = self.rng.permutation(self.n_batches)
            self.grad_off = self.rng.randint(self.n_offset)

        offset_value = self.grad_off + self.grad_perm[self.posg] * self.bs
        self.offset.set_value(offset_value)
        if self.unlabled > 0:
            self.posng = (self.posng + 1) % self.nat_batches
            if self.posng == 0:
                self.nat_perm = self.rng.permutation(self.nat_batches)
                self.nat_off = self.rng.randint(self.nat_offset)
            if self.unlabled == 1:
                offset_value = self.nat_off +  self.nat_perm[self.posng]*self.mbs
                self.m_offset.set_value(offset_value)
            else:
                offset = self.grad_off + self.pv_perm[self.posg] * self.bs
                self.nat_values[:self.bs] = self.data_x[offset:offset + self.bs]
                offset_value = self.nat_off +\
                        self.nat_perm[self.posng]*(self.mbs-self.bs)
                self.nat_values[self.bs:] = self.unlabled_data[
                    offset_value: offset_value+self.mbs - self.bs]
                self._natgrad.set_value(self.nat_values, borrow=True)


    def update_before_computing_natural_gradients(self):
        pass

    def update_before_evaluation(self):
        pass

