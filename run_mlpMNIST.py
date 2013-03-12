from model_mlpMNIST_standard import mlp
from SGD import SGD
from natSGD import natSGD
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
    model = mlp(state, data)
    if state['natSGD'] == 0:
        algo = SGD(model, state, data)
    else:
        algo = natSGD(model, state, data)
    main = MainLoop(data, model, algo, state, channel)
    main.main()

if __name__=='__main__':
    state = {}
    state['nhid'] = 1500
    state['cbs'] = 2500
    state['prefix'] = u'mlp_'
    state['seed'] = 203
    state['ebs'] = 2500
    state['overwrite'] = 1
    state['mbs'] = 2500
    state['bs'] = 2500
    state['mrtol'] = 0.0001
    state['matrix'] = u'KL'
    state['damp'] = 2.
    state['natSGD'] = 1
    state['lr'] = .5
    state['miters'] = 30
    state['loopIters'] = 2000
    state['timeStop'] = 30
    state['minerr'] = 1e-5
    state['adapt'] = 1
    state['profile'] = 0
    state['unlabled'] = 0
    state['validFreq'] = 10

    state['path'] = u'mnist.npz'
    state['trainFreq'] = 1
    state['minresQLP'] = 1
    state['trancond'] = 0.001
    state['saveFreq'] = 2
    state['lr'] = 1.
    state['lr0'] = 1.
    state['lr_beta'] = 20
    state['lr_adapt'] = 2
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

