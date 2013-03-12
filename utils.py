import numpy

import theano
import theano.tensor as TT


def safe_clone(cost, old_vars, new_vars):
    dummy_params = [x.type() for x in old_vars]
    dummy_cost = theano.clone(cost,
                              replace=zip(old_vars, dummy_params))
    return theano.clone(dummy_cost,
                        replace=zip(dummy_params, new_vars))


class forloop(theano.gof.Op):
    def __init__(self, loc_fn, n_steps, args, outputs):
        self.loc_fn = loc_fn
        self.n_steps = n_steps
        self.inputs = args
        self.outputs = outputs
        self.reset = theano.function([], [],
            updates=[(x, TT.zeros_like(x)) for x in self.outputs])

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) == type(other)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, *args):
        return theano.gof.Apply(self, args, [x.type() for x in args])

    def perform(self, node, inputs, outputs):
        for out in self.outputs:
            out.container.storage[0][:] = 0

        for inp, inp_var in zip(inputs, self.inputs):
            inp_var.set_value(inp, borrow=False)

        for step in xrange(self.n_steps):
            self.loc_fn(step)
        for mem, out in zip(outputs, self.outputs):
            mem[0] = out.get_value(return_internal_type=True, borrow=True)



if theano.sandbox.cuda.cuda_available:
    from theano.gof import local_optimizer
    from theano.sandbox.cuda.opt import register_opt
    from theano.sandbox.cuda.basic_ops import gpu_from_host, host_from_gpu
    from theano.sandbox.cuda.type import CudaNdarrayType

    @register_opt()
    @local_optimizer([])
    def local_gpu_forloop(node):
        if isinstance(node.op, forloop):
            sw = False
            for inp in node.inputs:
                if inp.owner and inp.owner.op == host_from_gpu:
                    sw = True
            if sw:
                inps = node.inputs
                nw_inps = []
                for inp in inps:
                    if not isinstance(inp.type, CudaNdarrayType):
                        nw_inps.append(gpu_from_host(inp))
                    else:
                        nw_inps.append(inp)
                new_outs = node.op(*nw_inps)
                return [host_from_gpu(x) for x in new_outs]
            else:
                return False


def print_time(secs):
    if secs < 120.:
        return '%6.3f sec' % secs
    elif secs <= 60 * 60:
        return '%6.3f min' % (secs / 60.)
    else:
        return '%6.3f h  ' % (secs / 3600.)


def print_mem(context=None):
    if theano.sandbox.cuda.cuda_enabled:
        rvals = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
        # Avaliable memory in Mb
        available = float(rvals[0]) / 1024. / 1024.
        # Total memory in Mb
        total = float(rvals[1]) / 1024. / 1024.
        if context == None:
            print ('Used %.3f Mb Free  %.3f Mb, total %.3f Mb' %
                   (total - available, available, total))
        else:
            info = str(context)
            print (('GPU status : Used %.3f Mb Free %.3f Mb,'
                    'total %.3f Mb [context %s]') %
                    (total - available, available, total, info))


def const(value):
    return TT.constant(numpy.asarray(value, dtype=theano.config.floatX))


def softmax(x):
    e = TT.exp(x)
    return e / TT.sum(e, axis=1).dimshuffle(0, 'x')
