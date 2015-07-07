# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 17:00:18 2015

@author: Eric Dodds

original MATLAB code by:
SAILnet: Sparse And Independent Local network _/)
Joel Zylberberg
UC Berkeley Redwood Center for Theoretical Neuroscience
joelz@berkeley.edu
Dec 2010

for work stemming from use of this code, please cite
Zylberberg, Murphy & DeWeese (2011) "A sparse coding model with synaptically
local plasticity and spiking neurons can account for the diverse shapes of V1
simple cell receptive fields", PLoS Computational Biology 7(10).
"""

import math
from math import ceil
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA

# TODO: functionality for using PC projections as input but still displaying
# RFs or STRFs

class SAILnet:
    """
    Runs SAILnet: Sparse And Independent Local Network
    Currently supports images and spectrograms. (Still working on spectrograms)
    """
    
    def __init__(self,
                 imagefilename = "IMAGES.mat",
                 imagevarname = "IMAGES",
                 datatype = "image",
                 timepoints = 25,
                 batch_size = 100,
                 niter = 50,
                 buffer = 20,
                 ninput = 256,
                 nunits = 256,
                 p = 0.05,
                 alpha = 1.,
                 beta = 0.01,
                 gamma = 0.1,
                 eta_ave = 0.3,
                 picklefile = 'SAILnetparams.pickle',
                 pca = None):
        """
        Create SAILnet object with given parameters. 
        Defaults are as used in Zylberberg et al.
        
        imagefile:          .mat file containing images for analysis
        datatype:           type of data in imagefile. Images and spectrograms
                            are supported. For PC projections representing
                            spectrograms, use spectro and input the relevant 
                            sklearn PCA object.
        timepoints:         number of time points in each spectrogram (ignored for images)
        batch_size:         number of image presentations between each learning step
        niter:              number of time steps in calculation of activities
                            for one image presentation
        buffer:             buffer on image edges
        ninput:             number of input units: for images, number of pixels
        nunits:             number of output units
        p:                  target firing rate
        alpha:              learning rate for inhibitory weights
        beta:               learning rate for feedforward weights
        gamma:              learning rate for thresholds
        eta_ave:            rate parameter for computing moving averages to get activity stats
        pca:                sklearn PCA object used to create vector inputs.
                            Used here to reconstruct spectrograms for display.
        """
        # Load input data from MATLAB file
        imagefile = scipy.io.loadmat(imagefilename)
        self.images = imagefile[imagevarname]
        self.datatype = datatype
        if self.datatype == "spectro" and self.images.shape[0] != ninput:
            # If the array is passed in with the indices swapped, transpose it
            self.images = np.transpose(self.images)
        if datatype != "image" and datatype != "spectro":
            input("Specified data type not supported. Supported types are image \
            and spectro. For PC vectors representing spectrograms, input the \
            sklearn PCA object used to create the PC projections. \
            Press any key to continue anyway.")

        # Store instance variables
        self.buffer = buffer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta_ave = eta_ave
        self.p = p        
        self.batch_size = batch_size
        self.niter = niter
        self.nunits = nunits # M in original MATLAB code
        self.picklefile = picklefile
        self.pca = pca
        
        # size and number of inputs
        if self.pca is None:
            # Then each input is 2D
            [self.imsize, _, self.nimages] = self.images.shape
        else:
            # Then we're using PC vectors, so each input is 1D
            [self.imsize, self.nimages] = self.images.shape
        
        # size of patches
        self.ninput = ninput # N in original MATLAB code
        if self.datatype == "spectro":
            self.lpatch = timepoints 
        else:
            self.lpatch = math.floor(math.sqrt(self.ninput))
                
        # initialize network parameters
        # Q are feedfoward weights (i.e. from input units to output units)
        # W are horizontal conections (among 'output' units)
        # theta are thresholds for the LIF neurons
        self.Q = np.random.randn(self.nunits, self.ninput)
        normmatrix = np.diag(1/np.sqrt(np.sum(self.Q*self.Q,1))) 
        self.Q = np.dot(normmatrix,self.Q) # normalize initial feedforward weight vectors
        self.W = np.zeros((self.nunits, self.nunits))
        self.theta = 2*np.ones(self.nunits)        
        
        # initialize average activity stats
        self.meanact_ave = self.p
        self.corrmatrix_ave = self.p**2
  
      
    def compute_activities(self, X):
        """
        Simulate LIF neurons to get spike counts
        function Y=activities(X,Q,W,theta)
        X:        input array
        Q:        feedforward weights
        W:        horizontal weights
        theta:    thresholds
        Y:        outputs
        """
        
        # rate parameter for numerical integration
        eta = 0.1
        
        # projections of stimuli onto feedforward weights
        B = np.dot(self.Q,X)
        
        # initialize values. Note that I've renamed some variables compared to 
        # Zylberberg's code. My variable names more closely follow the paper instead.
        u = np.zeros((self.nunits, self.batch_size)) # internal unit variables
        y = np.zeros((self.nunits, self.batch_size)) # external unit variables
        acts = np.zeros((self.nunits, self.batch_size)) # counts of total firings
        
        for t in range(self.niter):
            # DE for internal variables
            u = (1.-eta)*u + eta*(B - np.dot(self.W,y))
            
            # external variables should spike when internal variables cross threshholds
            y = np.array([u[:,ind] >= self.theta for ind in range(self.batch_size)])
            y = np.transpose(y)
            
            # add spikes to counts
            acts = acts + y
            
            # reset the internal variables of the spiking units
            u = u*(1-y)
            
        return acts  
        
    def showrfs(self):
        """Plot receptive fields."""
        if self.pca is not None:
            ffweights = self.pca.inverse_transform(self.Q)
        else:
            ffweights = self.Q
        M = self.nunits
        # length and height of each individual RF
        length = self.lpatch
        height = int(self.ninput/length)
        buf = 1 # buffer pixel(s) between RFs
        
        # n and m are number of rows and columns of spectrograms in the array
        if math.floor(math.sqrt(M))**2 != M:
            n = ceil(math.sqrt(M/2.))
            m = ceil(M/n)
        else:
            # M is a perfect square
            m = int(math.sqrt(M))
            n = m
            
        array = 0.5*np.ones((buf+m*(length+buf), buf+n*(height+buf)))
        k = 0
        
        # TODO: make this less hideously ugly
        # Right now it loops over every pixel in the array of STRFs.
        for j in range(m):
            for i in range(n):
                if k <= M:
                    clim = max(abs(self.Q[k,:]))
                    for li in range(height):
                        for lj in range(length):
                            array[buf+(j-1)*(length+buf)+lj, buf+(i-1)*(height+buf)+li] = \
                            ffweights[k,li+height*lj]/clim
                k = k+1
        
        arrayplot = plt.imshow(array,interpolation='nearest', aspect='auto')
        if self.datatype == "spectro":
            arrayplot.set_cmap("jet") 
        else:
            arrayplot.set_cmap("gray")
    
    def show_network(self):
        """
        Plot current values of weights, thresholds, and time-average firing
        correlations.
        """
        
        plt.subplot(2,2,1)
        plt.imshow(self.W)
        plt.colorbar()
        plt.title("Inhibitory weights")
        
        plt.subplot(2,2,2)
        C = self.corrmatrix_ave - \
            np.dot(self.meanact_ave,np.transpose(self.meanact_ave))
        plt.imshow(C)
        plt.colorbar()
        plt.title("Moving time-averaged correlation")
        
        plt.subplot(2,2,3)
        self.showrfs()
        plt.title("Feedforward weights")
        
        plt.subplot(2,2,4)
        plt.bar(np.arange(self.theta.size),self.theta) 
        plt.title(r"Thresholds $\theta$")
        
        plt.show()

    def randpatches(self):
        """
        Select random patches from the image data. Returns data array of
        batch_size columns, each of which is an unrolled image patch of size
        ninput.
        """
        # extract subimages at random from images array to make data array X
        X = np.zeros((self.ninput,self.batch_size))
        for i in range(self.batch_size):
                row = self.buffer + ceil((self.imsize- 
                self.lpatch-2*self.buffer)*np.random.rand())
                col = self.buffer + ceil((self.imsize- 
                self.lpatch-2*self.buffer)*np.random.rand())
                animage = self.images[row:row+self.lpatch,
                                      col:col+self.lpatch,
                                      math.floor(self.nimages*np.random.rand())]                     
                animage = animage.reshape(self.ninput)
                animage = animage - np.mean(animage)
                animage = animage/np.std(animage)
                X[:,i] = animage
        return X
        
    def randspectros(self):
        """
        Select random spectrograms from the spectrogram data. Returns array
        of batch_size columns, each of which is an unrolled image of a 
        spectrogram of size ninput.
        """
        X = np.zeros((self.ninput,self.batch_size))
        for i in range(self.batch_size):
            whichimage = math.floor(self.nimages*np.random.rand())
            animage = self.images[:,:,whichimage]
            animage = animage.reshape(self.ninput)
            animage = animage - np.mean(animage)
            animage = animage/np.std(animage)
            X[:,i] = animage
        return X
        
    def randvecs(self):
        """Select random vector inputs. Return an array of batch_size columns,
        each of which is an input vector. """
        X = np.zeros((self.ninput,self.batch_size))
        for i in range(self.batch_size):
            whichvec = math.floor(self.nimages*np.random.rand())
            X[:,i] = self.images[:, whichvec]
        return X
    
    def run(self, ntrials = 25000):
        """
        Run SAILnet for ntrials: for each trial, create a random set of image
        patches, present each to the network, and update the network weights
        after each set of batch_size presentations. Occasionally display progress.
        """
        for t in range(ntrials):
            # make data array X from random pieces of total data
            if self.pca is not None:
                X = self.randvecs()
            elif self.datatype == "spectro":
                X = self.randspectros()
            elif self.datatype == "image":
                X = self.randpatches()
            
            # compute activities for this data array
            acts = self.compute_activities(X)
            
            # compute statistics for this batch
            meanact = np.mean(acts,1)
            corrmatrix = np.dot(acts, np.transpose(acts))
            
            # update lateral weights with Foldiak's rule 
            # (inhibition for decorrelation)
            dW = self.alpha*(corrmatrix - self.p**2)
            self.W = self.W + dW
            self.W = self.W - np.diag(np.diag(self.W)) # zero diagonal entries
            self.W[self.W < 0] = 0 # force weights to be inhibitory
            
            # update feedforward weights with Oja's rule
            sumsquareacts = np.sum(acts*acts,1) # square, then sum over images
            dQ = np.dot(acts,np.transpose(X)) - \
                np.dot(np.diag(sumsquareacts), self.Q)
            self.Q = self.Q + self.beta*dQ/self.batch_size
            
            # update thresholds with Foldiak's rule: keep firing rates near target
            dtheta = self.gamma*(np.sum(acts,1)/self.batch_size - self.p)
            self.theta = self.theta + dtheta
            
            # compute moving averages of meanact and corrmatrix
            self.meanact_ave = (1 - self.eta_ave)*self.meanact_ave + \
                self.eta_ave*meanact
            self.corrmatrix_ave = (1 - self.eta_ave)*self.corrmatrix_ave + \
                self.eta_ave*corrmatrix
                
            # display network state and activity statistics every 50 trials
            if t % 50 == 0:
                plt.figure(1)
                plt.clf()
                self.show_network()
                plt.figure(2)
                plt.clf()
                self.showrfs()
                print("Trial number: " + str(t))
                if t % 5000 == 0:
                    # save progress
                    print("Saving progress...")
                    self.save_params()
                    print("Done. Continuing to run...")
                    
        # Save final parameter values            
        self.save_params()              
     
    
    def save_params(self, filename=None):
        """
        Save parameters to a pickle file, to be picked up later. By default
        we save to the file name stored with the SAILnet instance, but a different
        file can be passed in as the string filename.
        """
        if filename is None:
            filename = self.picklefile
        with open(filename,'wb') as f:
            pickle.dump([self.Q, self.W, self.theta, 
                         self.meanact_ave, self.corrmatrix_ave], f)

    def load_params(self, filename = None):
        """Load parameters (e.g., weights) from a previous run from pickle file."""
        if filename is None:
            filename = self.picklefile
        with open(filename, 'rb') as f:
            self.Q, self.W, self.theta, self.meanact_ave, self.corrmatrix_ave = \
            pickle.load(f)               
            
            