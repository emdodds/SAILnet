import numpy as np
import scipy.io as io
import pickle
import argparse
import os
import sys
import StimSet
sys.path.append('../Code/')
import SAILmods
import SAILnet


parser = argparse.ArgumentParser()
parser.add_argument('--scaled', dest='scaled', action='store_true')
parser.add_argument('--not-scaled', dest='scaled', action='store_false')
parser.add_argument('-g', '--gain', default=2, type=float)
parser.add_argument('--load', dest='load', action='store_true')
parser.add_argument('-d', '--data', default='images', type=str)
parser.add_argument('-p', '--firing_rate', default=0.05, type=float)
parser.add_argument('--oc', default=1, type=float)
parser.add_argument('--keep_only_error', dest='keep_only_error', action='store_true')
parser.set_defaults(keep_only_error=False)
parser.set_defaults(scaled=True)
parser.set_defaults(load=False)
args = parser.parse_args()
datatype = args.data

kwargs = {'p': args.firing_rate,
          'theta0': 1.5}
if args.scaled:
    Net = SAILmods.VarTimeSAILnet
    prefix = 'scaled' + str(args.gain)
    kwargs['gain_rate'] = 0.0
    kwargs['gain'] = args.gain
else:
    Net = SAILnet.SAILnet
    prefix = ''
paramfile = prefix+'SAIL'+str(args.oc)+'oc'+str(args.firing_rate)+'p.pickle'

if datatype == 'images':
    wholeims = io.loadmat('../../vision/Data/IMAGES_vh.mat')['IMAGES']
    wholeims /= wholeims.std()
    numinput = 256
    numunits = int(numinput*args.oc)
    net = Net(data=wholeims, nunits=numunits,
              paramfile='bvh'+paramfile,
              **kwargs)
elif datatype == 'fieldraw':
    wholeims = io.loadmat('../../vision/Data/IMAGES_RAW.mat')['IMAGESr']
    wholeims /= wholeims.std()
    numinput = 256
    numunits = int(numinput*args.oc)
    net = Net(data=wholeims, nunits=numunits,
              paramfile='fraw'+paramfile,
              **kwargs)
    net.stims.patchwisenorm = True
elif datatype == 'pcaimages':
    datafile = '../../vision/Data/300kvanHateren'
    numinput = 200
    with open(datafile+'PCA', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile+'200.npy')
    data = data/data.std()
    numunits = int(numinput*args.oc)
    net = Net(data=data, nunits=numunits,
              datatype='image',
              pca=mypca,
              stimshape=origshape, ninput=numinput,
              paramfile='pca80vh'+paramfile,
              **kwargs)
elif datatype == 'smallpcaimages':
    datafile = '../../vision/Data/bvh_16x16_PCAd.npy'
    numinput = 200
    with open('../../vision/Data/bvh_16x16_PCA.pickle', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile)[:, :numinput]
    mypca.dim = numinput
    data /= data.std()
    numunits = int(numinput * args.oc)
    net = Net(data=data, nunits=numunits,
              datatype='image',
              pca=mypca,
              stimshape=origshape, ninput=numinput,
              paramfile='pca16vh'+paramfile,
              **kwargs)
elif datatype.startswith('vh32_256ds'):
    datafile = '../../vision/Data/vh32_256.npy'
    numinput = 256
    with open('../../vision/Data/vh32_256PCA.pickle', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile)[:, :numinput]
    mypca.dim = numinput
    power = float(datatype.split('ds')[-1])
    data = data.dot(np.diag(np.power(mypca.sValues[:numinput], power)))
    data /= data.std()
    numunits = int(numinput * args.oc)
    net = Net(data=data, nunits=numunits,
              datatype='image',
              pca=mypca,
              stimshape=origshape, ninput=numinput,
              paramfile=datatype+paramfile,
              **kwargs)
elif datatype == 'spectro':
    datafile = '../../audition/Data/allTIMIT'
    numinput = 200
    with open(datafile+'_pca.pickle', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile+'.npy')
    data = data/data.std()
    numunits = int(numinput*args.oc)
    net = Net(data=data, nunits=numunits,
              datatype='spectro',
              pca=mypca,
              stimshape=origshape, ninput=numinput,
              paramfile='allTIMIT'+paramfile,
              **kwargs)
else:
    raise ValueError('Data type not recognized: ' + str(datatype))

if args.load:
    net.load(net.paramfile)
#else:
#    net.set_dot_inhib()

#net.run(10000)
#net.save()
net.run(100000)#, rate_decay=0.99999)
net.save()

if args.keep_only_error:
    error = np.mean(net.errorhist[-1000:])
    np.save('hyperparams/'+net.paramfile+'mse', error)
    os.remove(net.paramfile)
