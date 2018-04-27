# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:33:44 2015

@author: Eric
"""

from os import chdir
chdir("..")

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from pca.pca import PCA
import SAILnet

imfile = "patches.mat"
imname = "patches"

nullpca = PCA()
normalpca = PCA()
whiteningpca = PCA(whiten=True)

patches = scipy.io.loadmat(imfile)[imname]
origshape = patches.shape
veclength = origshape[0]*origshape[1]
nimages = origshape[2]         

# unroll images
patches = patches.reshape((veclength,nimages))

scipy.io.savemat("unrolledpatches.mat",{"patches":patches})

# the PCA object wants each row to be a data point by default
patches = np.transpose(patches) 

normalpca.fit(patches)
whiteningpca.fit(patches)

transpatches = normalpca.transform(patches)
# whiten, then return to original rep
whitepatches = whiteningpca.transform_zca(patches)

dumbnet = SAILnet.SAILnet()
dumbnet.Q = whitepatches[:dumbnet.Q.shape[0],:]
dumbnet.showrfs()

whitepatches = whitepatches.T
scipy.io.savemat("zcapatches.mat",{"patches":whitepatches})

nullpca.fit(transpatches)
nullpca.eVectors = nullpca.eVectors**2

plt.figure(1)
x = np.arange(veclength)
plt.plot(x, normalpca.sValues, 'bo', x, nullpca.sValues, 'gx')

net = SAILnet.SAILnet(imagefilename = "unrolledpatches.mat",
                      imagevarname = imname, datatype = "image",
                      picklefile = "pcareptestrun.pickle", pca = nullpca)
#net = SAILnet.SAILnet(imagefilename = "zcapatches.mat",
#                      imagevarname = imname, datatype = "image",
#                      picklefile = "zcatestrun.pickle", pca = whiteningpca)                      
#                      
#net.run()