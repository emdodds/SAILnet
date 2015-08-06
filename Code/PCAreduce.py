# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:37:27 2015

@author: Eric Dodds

Read in images, find principle components, write out images reconstructed
from subset of the PCs.
The PCA class is written by Jesse Livezey.
"""

import numpy as np
import scipy.io
from pca.pca import PCA # Jesse's PCA class
import pickle

def preprocessimages(filename = "../../audition/speechdata.mat", 
                     dataname = "speechdata0",
                     ncomponents=200, 
                     outfilename = "processedspeech.mat",
                     outname = 'processedspeech',
                     pcafilename = 'pickledpca.pickle',
                     whiten = True):
    """
    Reads images from given .mat file, whitens and reduces dimensionality by
    PCA, retaining ncomponents PCs. Returns processed images and the
    PCA object used. This same object should be used to do the inverse transform.
    """
    
    data = scipy.io.loadmat(filename)[dataname]
    origshape = data.shape
    veclength = origshape[0]*origshape[1]
    nimages = origshape[2]         
    
    # unroll images
    data = data.reshape((veclength,nimages))
    # the PCA object wants each row to be a data point
    data = np.transpose(data) 
    
    # This stops NaN/inf from showing up later and crashing things. I don't
    # think the nan_to_num actually does anything since the data should only
    # have NaNs if the psd in MATLAB was exactly 0 at some point. But I'm being
    # safe.
    data = np.nan_to_num(data)
    data = np.clip(data,-1000,1000)   
    
    # center and normalize. This may be unnecessary since the PCA object also does it?
    datamean = np.mean(data,axis=0)
    data = data - datamean
    datastd = np.std(data,axis=0)
    data = data/datastd
    
    # We want to both reduce dimensionality and whiten
    pca = PCA(dim = ncomponents, whiten=whiten)
    pca.fit(data)
    data = pca.transform(data)  
    
    data = np.transpose(data)
    # save data and pca object
    scipy.io.savemat(outfilename,{outname : data})
    with open(pcafilename,'wb') as f:
        pickle.dump([pca, origshape, datamean, datastd], f)
    
    return data, pca, origshape, datamean, datastd
        
def restoreIm(transformeddata, pca, origshape, datamean, datastd):
    """Given a PCA object and transformeddata that consists of projections onto
    the PCs, return images by using the PCA's inverse transform and reshaping to
    the provided origshape."""
    if transformeddata.shape[0] < transformeddata.shape[1]:
        transformeddata = np.transpose(transformeddata)
    data = pca.inverse_transform(transformeddata)
    # restore the shape and scale of the data before plotting
    data = data*datastd
    data = data + datamean
    data = np.transpose(data)
    return data.reshape(origshape)
    
def restoreImfromfile(datafile = "processedspeech.mat", 
                      dataname = "processedspeech",
                      pcafilename = 'pickledpca.pickle'):
    data = scipy.io.loadmat(datafile)[dataname]
    with open(pcafilename,'rb') as f:
        pca, origshape, datamean, datastd = pickle.load(f)
    return restoreIm(data, pca, origshape, datamean, datastd)
    
#transformeddata, pca, origshape, datamean, datastd = preprocessimages(outfilename = "processedspeechnowhiten.mat", whiten = False)
#reconst = restoreIm(transformeddata, pca, origshape)
#reconst = restoreImfromfile((256, 25, 32601))
#plt.figure()
#plt.imshow(reconst[:,:,22222], interpolation='nearest', aspect='auto')
#plt.gca().invert_yaxis()

    