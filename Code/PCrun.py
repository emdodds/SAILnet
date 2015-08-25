# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:52:50 2015

@author: Eric
"""

import SAILnet
import pickle
import scipy.io

ntimes = 25
nfreqs = 256
overcompleteness = 4
numinput = 200
numunits = int(overcompleteness*numinput)
with open("../Pickles/spectropca.pickle",'rb') as f:
        pca, origshape, datamean, datastd = pickle.load(f)
spectros = scipy.io.loadmat("../Data/processedspeech.mat")["processedspeech"]
net = SAILnet.SAILnet(images = spectros, datatype = "spectro",
                      imshape=(ntimes,nfreqs), ninput = numinput, nunits = numunits,
                      picklefile = '../Pickles/dummy.pickle',
                      pca = pca)
                      
#net.run()