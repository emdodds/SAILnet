# -*- coding: utf-8 -*-
"""
@author: Eric
"""

import SAILnet
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Learn dictionaries for LCA with given parameters.")
parser.add_argument('-o', '--overcompleteness', default=4, type=float)
parser.add_argument('-d', '--datafile', default='../../audition/Data/speech_ptwisecut', type=str)
parser.add_argument('-r', '--resultsfolder', default='../../audition/Results/',type=str)
parser.add_argument('-s', '--datasuffix', default='ptwise', type=str)
parser.add_argument('-p', '--p', default=0.05, type=float)
args=parser.parse_args()

datafile = args.datafile
resultsfolder = args.resultsfolder
oc = args.overcompleteness
datasuffix = args.datasuffix
p = args.p

numinput = 200
numunits = int(oc*numinput)

with open(datafile+'_pca.pickle', 'rb') as f:
    mypca, origshape = pickle.load(f)
data = np.load(datafile+'.npy')*200

net = SAILnet.SAILnet(data=data, ninput = numinput, nunits=numunits, datatype="spectro", pca = mypca,  stimshape=origshape, picklefile='dummy')

net.p=p

savestr = resultsfolder+'SAIL'+str(oc)+'OC' + str(p) + datasuffix
net.save_params(savestr+'.pickle')
net.run(ntrials=50000)
net.run(ntrials=200000, rate_decay=.99995)
net.sort_dict()
net.save_params()
