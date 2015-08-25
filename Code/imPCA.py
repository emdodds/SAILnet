# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:26:01 2015

@author: Eric
"""

#import pickle
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import PCAreduce

#filename = "../../audition/speechdata.mat"
#dataname = "speechdata0"

filename = "patches.mat"
dataname = "patches"
ncomponents = None
outfilename = "processedimages.mat"
outname = "processedimages"
pcafilename = "imagepca.pickle"
whiten = True

transformed, pca, origshape, datamean, datastd = PCAreduce.preprocessimages(filename,
                                                                            dataname,
                                                                            ncomponents,
                                                                            outfilename,
                                                                            outname,
                                                                            pcafilename,
                                                                            whiten)

images = scipy.io.loadmat(filename)[dataname]

    
# plot some random unprocessed images    
plt.figure()
whichspect = np.floor(np.random.rand(4)*images.shape[-1])
for i in range(4):
    plt.subplot(2,4,i+1)
    plt.imshow(images[:,:,whichspect[i]],cmap='gray',interpolation='nearest', aspect='auto')
    plt.gca().invert_yaxis()
    
#del spects # clear up memory    
    
# plot the corresponding reconstructed images
reconst = PCAreduce.restoreIm(transformed, pca, origshape, datamean, datastd)
for i in range(4):
    plt.subplot(2,4,i+5)
    plt.imshow(reconst[:,:,whichspect[i]],cmap='gray',interpolation='nearest', aspect='auto')
    plt.gca().invert_yaxis()    