from model_convTFD_slow import convMat
from SGD import SGD
from natSGD import natSGD
from mainLoop import MainLoop
from dataTFD_standard import DataTFD_48
import numpy

def jobman(state, channel):
    state['path'] = '/data/lisa/data/faces/TFD/'
    #state['path'] = '/RQexec/pascanur/data/'
    rng = numpy.random.RandomState(state['seed'])
    data = DataTFD_48(state['path'], state['mbs'], state['bs'], rng, fold=state['fold'],
                     unlabled=state['unlabled'])
    model = convMat(state, data)
    if state['natSGD'] == 0:
        algo = SGD(model, state, data)
    else:
        algo = natSGD(model, state, data)

    main = MainLoop(data, model, algo, state, channel)
    main.main()

if __name__=='__main__':
    state = {}

    state['fold'] = 0
    state['unlabled'] = 1
    #state['path'] = '/scratch/pascanur/data/'
    state['path'] = '/data/lisa/data/faces/TFD/'
    #state['path'] = '/RQexec/pascanur/data/'

    state['mbs'] = 128*3
    state['bs']  = 128*2
    state['ebs'] = 128*2
    state['cbs'] = 128

    state['matrix'] = 'KL'
    state['natSGD'] = 1


    state['loopIters'] = 6000
    state['timeStop'] = 32*60
    state['minerr'] = 1e-5

    state['lr'] = .2
    state['lr_adapt'] = 0

    state['damp'] = 5.
    state['adapt'] = 1.
    state['mindamp'] = .5
    state['damp_ratio'] =5./4.
    state['minerr'] = 1e-5

    state['seed'] = 123


    state['profile'] = 0
    state['minresQLP'] = 1
    state['mrtol'] = 1e-4
    state['miters'] = 50
    state['trancond'] = 1e-4


    state['trainFreq'] = 1
    state['validFreq'] = 20
    state['saveFreq'] = 20

    state['prefix'] = 'conv_'
    state['overwrite'] = 1
    jobman(state, None)

