import numpy as np
import scipy.io as io
import pickle
import argparse
import sys
sys.path.append('../Code/')
import SAILmods

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='images', type=str)
args = parser.parse_args()
datatype = args.data

oc = 6

if datatype == 'images':
    wholeims = io.loadmat('../../vision/Data/IMAGES_vh.mat')['IMAGES']
    wholeims /= wholeims.std()
    numinput = 256
    numunits = numinput*oc
    net = SAILmods.VarTimeSAILnet(data=wholeims, nunits=numunits, theta0=2.5,
                                  paramfile='scaledSAIL_bvh_6oc.pickle')
elif datatype == 'spectro':
    datafile = '../../audition/Data/allTIMIT'
    numinput = 200
    with open(datafile+'_pca.pickle', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile+'.npy')
    data = data/data.std()
    numunits = numinput = oc
    net = SAILmods.VarTimeSAILnet(data=data, nunits=numunits,
                                  datatype='spectro',
                                  pca=mypca,
                                  stimshape=origshape, ninput=numinput,
                                  theta0=2.5,
                                  paramfile='scaledSAIL_allTIMIT_6oc.pickle')

net.set_dot_inhib()
net.p = 0.01
net.beta = 0.001
net.save()

net.run(10000)
net.save()
net.run(100000, rate_decay=0.99999)
