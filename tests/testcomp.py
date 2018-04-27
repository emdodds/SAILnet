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

filename = "../../../audition/speechdata.mat"
dataname = "speechdata0"

spects = scipy.io.loadmat(filename)[dataname][:,:,:1000]
    
# plot some random unprocessed spectrograms    
plt.figure()
whichspect = np.floor(np.random.rand(4)*spects.shape[-1])
for i in range(4):
    plt.subplot(2,4,i+1)
    plt.imshow(spects[:,:,whichspect[i]],interpolation='nearest', aspect='auto')
    plt.gca().invert_yaxis()
    
del spects # clear up memory    
    
# plot the corresponding reconstructed spectrograms
reconst = PCAreduce.restoreImfromfile(datafile = "processedspeech", pcafilename = 'pickledpca.pickle')
for i in range(4):
    plt.subplot(2,4,i+5)
    plt.imshow(reconst[:,:,whichspect[i]],interpolation='nearest', aspect='auto')
    plt.gca().invert_yaxis()    