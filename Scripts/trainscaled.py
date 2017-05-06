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
parser.add_argument('-d', '--data', default='images', type=str)
parser.set_defaults(scaled=True)
args = parser.parse_args()
datatype = args.data
Net = SAILmods.VarTimeSAILnet if args.scaled else SAILnet.SAILnet
prefix = 'scaled' if args.scaled else ''

if datatype == 'images':
    oc = 8
    wholeims = io.loadmat('../../vision/Data/IMAGES_vh.mat')['IMAGES']
    wholeims /= wholeims.std()
    numinput = 256
    numunits = numinput*oc
    net = Net(data=wholeims, nunits=numunits, theta0=2.0,
              paramfile=prefix+'SAIL_bvh_8oc.pickle')
elif datatype == 'pcaimages':
    oc = 10
    datafile = '../../vision/Data/300kvanHateren'
    numinput = 200
    with open(datafile+'PCA', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile+'200.npy')
    data = data/data.std()
    numunits = numinput*oc
    net = Net(data=data, nunits=numunits, theta0=2.0,
              datatype='image',
              pca=mypca,
              stimshape=origshape, ninput=numinput,
              paramfile=prefix+'SAIL_pcavh_10oc.pickle')

elif datatype == 'spectro':
    oc = 10
    datafile = '../../audition/Data/allTIMIT'
    numinput = 200
    with open(datafile+'_pca.pickle', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile+'.npy')
    data = data/data.std()
    numunits = numinput*oc
    net = Net(data=data, nunits=numunits, theta0=2.0,
              datatype='spectro',
              pca=mypca,
              stimshape=origshape, ninput=numinput,
              paramfile=prefix+'scaledSAIL_allTIMIT_10oc.pickle')

net.set_dot_inhib()
net.p = 0.01
net.beta = 0.0
net.run(2000)
net.beta = 0.001
net.save()

net.run(10000)
net.save()
net.run(100000, rate_decay=0.99999)
