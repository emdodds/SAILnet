# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:12:36 2015

@author: Eric
"""

#import pickle
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import PCAreduce

filename = "../../../audition/speechdata.mat"
dataname = "speechdata0"

ncomponents = 400
outfilename = "../Data/processedspeech400.mat"
outname = "processedspeech"
pcafilename = "../Pickles/spectropca400.pickle"
whiten = True

transformed, pca, origshape, datamean, datastd = PCAreduce.preprocessimages(filename,
                                                                            dataname,
                                                                            ncomponents,
                                                                            outfilename,
                                                                            outname,
                                                                            pcafilename,
                                                                            whiten)

raw = scipy.io.loadmat(filename)[dataname]

plt.figure(1)
# plot some random raw spectrograms
whichspect = np.floor(np.random.rand(4)*raw.shape[-1])
for i in range(4):
    plt.subplot(2,4,i+1)
    plt.imshow(raw[:,:,whichspect[i]],cmap='gray',interpolation='nearest', aspect='auto')
    plt.gca().invert_yaxis()
    
#del spects # clear up memory    
    
# plot the corresponding reconstructed images
reconst = PCAreduce.restoreIm(transformed, pca, origshape, datamean, datastd)
for i in range(4):
    plt.subplot(2,4,i+5)
    plt.imshow(reconst[:,:,whichspect[i]],cmap='gray',interpolation='nearest', aspect='auto')
    plt.gca().invert_yaxis()  
    
print ("Variance explained: " + str(pca.explained_variance()))