'''
Please use the branch
https://github.com/pascanur/Theano/tree/downsampling_rop
'''
from model_convMNIST_standard import convMat
from natSGD import natSGD
from SGD import SGD
from mainLoop import MainLoop
from dataMNIST_standard import DataMNIST
import numpy

def jobman(state, channel):
    rng = numpy.random.RandomState(state['seed'])
    data = DataMNIST(
        state['path'],
        state['mbs'],
        state['bs'],
        rng,
        state['unlabled'])
    model = convMat(state, data)
    if state['natSGD'] == 0:
        algo = SGD(model, state, data)
    else:
        algo = natSGD(model, state, data)

    main = MainLoop(data, model, algo, state, channel)
    main.main()

if __name__=='__main__':
    state = {}
    state['path'] = 'mnist.npz'
    state['mbs'] =10000
    state['bs'] = 10000
    state['adapt'] = 3
    state['ebs'] = 4000
    state['lr'] = 1.
    state['damp'] = 5.
    state['nkerns'] = '[32,64]'
    state['seed'] = 200
    state['cbs'] = 1000
    state['matrix'] = 'KL'
    state['natSGD'] = 1
    state['timeStop'] = 4
    state['nhid'] = 750
    state['unlabled'] =1
    state['minerr'] = 1e-4
    state['profile'] = 0
    state['prefix'] = 'Conv_'
    state['minresQLP'] = 1
    state['mrtol'] = 1e-4
    state['miters'] = 40
    state['trancond'] = 1e-3
    state['loopIters'] = 1000
    state['trainFreq'] = 1
    state['validFreq'] = 10
    state['saveFreq'] = 20
    state['overwrite'] = 1
    state['lr'] = 1.
    state['lr_adapt'] = 1
    state['lr_adapt_start'] = 150
    state['lr_adapt_change'] = 100
    state['lr_adapt_decrease'] = .5
    state['damp'] = 6.
    state['adapt'] = 1.
    state['adapt_start'] = 100
    state['adapt_change'] = 100
    state['adapt_decrease'] = .5
    state['minerr'] = 1e-5
    jobman(state, None)

