# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:23:27 2015

@author: Eric

Takes 512x512 pixel images as input, spits out 16x16 patches.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

filename = "IMAGES.mat"
dataname = "IMAGES"
newlength = 16

images = scipy.io.loadmat(filename)[dataname]

# remove artifacts around edges of images
buffer = 20 # as used in original SAILnet
images = images[buffer:-buffer,buffer:-buffer,:]


lengthratio = int(np.floor(images.shape[0]/newlength))
patchesperim = lengthratio**2
nimages = images.shape[-1]

patches = np.zeros((newlength, newlength, nimages*patchesperim))

for i in range(nimages):
    for j in range(lengthratio):
        for k in range(lengthratio):
            patches[:,:,patchesperim*i+lengthratio*j+k] = \
            images[j*newlength:(j+1)*newlength,k*newlength:(k+1)*newlength,i]
            
scipy.io.savemat("patches.mat",{"patches":patches})

# plot first image and a few patches in the upper left corner to verify that
# the slicing and dicing has worked properly
plt.figure(1)
origim = plt.imshow(images[:,:,0], interpolation = "nearest")
origim.set_cmap("gray")
plt.figure(2)
for i in range(6):
    for j in range(6):
        plt.subplot(6,6,6*i+j+1)
        patch = plt.imshow(patches[:,:,lengthratio*i+j], interpolation="nearest")
        patch.set_cmap("gray")
