# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:20:46 2015

@author: Eric
"""

import SAILnet
import pickle

ntimes = 25
nfreqs = 256
overcompleteness = 1
numinput = 200
numunits = int(overcompleteness*numinput)
with open("pickledpca.pickle",'rb') as f:
        pca = pickle.load(f)
net = SAILnet.SAILnet(imagefilename = "processedspeech.mat",
                      imagevarname = "processedspeech", datatype = "spectro",
                      ninput = numinput, nunits = numunits, pca = pca)
#net.run()
#net.load_params("params_speech2015_6_30.pickle")
#net.showrfs()