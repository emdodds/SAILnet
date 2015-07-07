# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:52:50 2015

@author: Eric
"""

import SAILnet
import pickle

ntimes = 25
nfreqs = 256
overcompleteness = 2
numinput = 200
numunits = int(overcompleteness*numinput)
with open("pickledpcanowhiten.pickle",'rb') as f:
        pca, origshape, datamean, datastd = pickle.load(f)
net = SAILnet.SAILnet(imagefilename = "processedspeechnowhiten.mat",
                      imagevarname = "processedspeech", datatype = "spectro",
                      timepoints = ntimes, ninput = numinput, nunits = numunits,
                      picklefile = 'SAILnetaudio.pickle',
                      pca = pca)
                      
net.run()