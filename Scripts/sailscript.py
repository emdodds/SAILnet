# -*- coding: utf-8 -*-
"""
@author: Eric
"""

import SAILnet
import pickle
import argparse
import numpy as np
import scipy.io as io

parser = argparse.ArgumentParser(description="Run SAILnet with given parameters.")
parser.add_argument('-o', '--overcompleteness', default=4, type=float)
parser.add_argument('-d', '--data', default='images', type=str)
parser.add_argument('-r', '--resultsfolder', default='',type=str)
parser.add_argument('-s', '--savesuffix', default='', type=str)
parser.add_argument('-p', '--p', default=0.05, type=float)
parser.add_argument('--load', action='store_true')
args=parser.parse_args()

data = args.data
resultsfolder = args.resultsfolder
oc = args.overcompleteness
p = args.p
savesuffix = args.savesuffix
load = args.load

if data == 'images':
    datafile = '../../vision/Data/IMAGES.mat'
    numinput = 256
    numunits = int(oc*numinput)
    data = io.loadmat(datafile)["IMAGES"]
    if resultsfolder == '':
        resultsfolder = '../../vision/Results/'
    net = SAILnet.SAILnet(data=data, nunits = numunits)
elif data == 'spectros':
    datafile = '../../audition/Data/speech_ptwisecut'
    numinput = 200
    numunits = int(oc*numinput)    
    with open(datafile+'_pca.pickle', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile+'.npy')
    data = data/data.std()
    if resultsfolder == '':
        resultsfolder = '../../audition/Results/'       
    net = SAILnet.SAILnet(data=data, ninput = numinput, nunits=numunits,
                          datatype="spectro", pca = mypca,  stimshape=origshape)

net.p=p

savestr = resultsfolder+'SAIL'+str(oc)+'OC' + str(p) + savesuffix
if load:
    net.load(savestr+'.pickle')
net.save(savestr+'.pickle')
net.run(ntrials=50000)
net.run(ntrials=200000, rate_decay=.99995)
net.save()
