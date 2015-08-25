# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:04:16 2015

@author: Eric
"""

import SAILnet
import pickle

with open("imagepca.pickle",'rb') as f:
        pca, origshape, datamean, datastd = pickle.load(f)
net = SAILnet.SAILnet(imagefilename = "processedimages.mat",
                      imagevarname = "processedimages", datatype = "image",
                      picklefile = 'SAILnetimPC.pickle',
                      pca = pca)
                      #alpha = 1, beta = 0.01, gamma = 0.1, theta0 = 2)
                      
net.run()