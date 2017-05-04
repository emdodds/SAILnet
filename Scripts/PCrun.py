# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:52:50 2015

@author: Eric
"""

import SAILnet
import pickle
#import scipy.io
import numpy as np
import pca.pca
import sys
sys.modules['pca'] = pca.pca

ntimes = 25
nfreqs = 256
overcompleteness = 2
numinput = 200
numunits = int(overcompleteness*numinput)
picklefile = '../../../audition/Data/spectropca2.pickle'
datafile = '../../../audition/Data/processedspeech2.npy'
#picklefile = "../Pickles/spectropca.pickle"
with open(picklefile,'rb') as f:
        pca, origshape, datamean, datastd = pickle.load(f)
#spectros = scipy.io.loadmat("../Data/processedspeech.mat")["processedspeech"]
spectros = np.load(datafile)
        
net = SAILnet.SAILnet(data = spectros, datatype = "spectro",
                      stimshape=origshape, ninput = numinput, nunits = numunits,
                      picklefile = '../Pickles/dummy.pickle',
                      pca = pca)

#net.save_params('../../../audition/Results/SAILnet10_8')
                      
#net.run(ntrials=50000)
#net.adjust_rates(.1)
#net.run(ntrials=100000)
#net.adjust_rates(.1)
#net.run(ntrials=1000000)