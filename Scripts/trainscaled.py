import numpy as np
import scipy.io as io
import pickle
import argparse
import sys
sys.path.append('../Code/')
import SAILmods
import SAILnet


parser = argparse.ArgumentParser()
parser.add_argument('--scaled', dest='scaled', action='store_true')
parser.add_argument('--not-scaled', dest='scaled', action='store_false')
parser.add_argument('--load', dest='load', action='store_true')
parser.add_argument('-d', '--data', default='images', type=str)
parser.set_defaults(scaled=True)
parser.set_defaults(load=False)
args = parser.parse_args()
datatype = args.data

kwargs = {'p': 0.01,
          'theta0': 2.5}
if args.scaled:
    Net = SAILmods.VarTimeSAILnet
    prefix = 'scaled'
    kwargs['gain_rate'] = 0.0
    kwargs['gain'] = 2.0
else:
    Ne = SAILnet.SAILnet
    prefix = ''

if datatype == 'images':
    oc = 8
    wholeims = io.loadmat('../../vision/Data/IMAGES_vh.mat')['IMAGES']
    wholeims /= wholeims.std()
    numinput = 256
    numunits = numinput*oc
    net = Net(data=wholeims, nunits=numunits,
              paramfile=prefix+'SAIL_bvh_8oc.pickle',
              **kwargs)
elif datatype == 'pcaimages':
    oc = 10
    datafile = '../../vision/Data/300kvanHateren'
    numinput = 200
    with open(datafile+'PCA', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile+'200.npy')
    data = data/data.std()
    numunits = numinput*oc
    net = Net(data=data, nunits=numunits,
              datatype='image',
              pca=mypca,
              stimshape=origshape, ninput=numinput,
              paramfile=prefix+'SAIL_pcavh_10oc.pickle',
              **kwargs)

elif datatype == 'spectro':
    oc = 10
    datafile = '../../audition/Data/allTIMIT'
    numinput = 200
    with open(datafile+'_pca.pickle', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile+'.npy')
    data = data/data.std()
    numunits = numinput*oc
    net = Net(data=data, nunits=numunits,
              datatype='spectro',
              pca=mypca,
              stimshape=origshape, ninput=numinput,
              paramfile=prefix+'scaledSAIL_allTIMIT_10oc.pickle',
              **kwargs)

if args.load:
    net.load(net.paramfile)
else:
    net.set_dot_inhib()

net.run(10000)
net.save()
net.run(100000, rate_decay=0.99999)
