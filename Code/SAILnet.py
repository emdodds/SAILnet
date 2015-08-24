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

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pickle
from DictLearner import DictLearner

class SAILnet(DictLearner):
    """
    Runs SAILnet: Sparse And Independent Local Network
    Currently supports images and spectrograms. (Still working on spectrograms)
    """
    
    def __init__(self,
                 images = None,
                 datatype = "image",
                 imshape = None,
                 batch_size = 100,
                 niter = 50,
                 buffer = 20,
                 ninput = 256,
                 nunits = 256,
                 p = 0.05,
                 alpha = 1.,
                 beta = 0.01,
                 gamma = 0.1,
                 theta0 = 2,
                 eta_ave = 0.3,
                 picklefile = 'SAILnetparams.pickle',
                 pca = None):
        """
        Create SAILnet object with given parameters. 
        Defaults are as used in Zylberberg et al.
        
        Args:
        imagefile:          .mat file containing images for analysis
        datatype:           type of data in imagefile. Images and spectrograms
                            are supported. For PC projections representing
                            spectrograms, use spectro and input the relevant 
                            PCA object. Images are assumed to be squares.
        imshape:            Shape of images/spectrograms. Square by default.
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
        theta0:             initial value of thresholds
        eta_ave:            rate parameter for computing moving averages to get activity stats
        picklefile:         File in which to save pickled parameters.
        pca:                PCA object used to create vector inputs.
                            Used here to reconstruct spectrograms for display.
                            
        Raises:
        ValueError when datatype is not one of the supported options.
        """
        # If no input data passed in, use IMAGES.mat
        if images is None:        
            self.images = scipy.io.loadmat("../Data/IMAGES.mat")["IMAGES"]
        else:
            self.images = images
        
        self.datatype = datatype
        if self.datatype == "spectro" and self.images.shape[0] != ninput:
            if self.images.shape[-1] == ninput:
                # If the array is passed in with the indices swapped, transpose it
                self.images = np.transpose(self.images)
            else:
                raise ValueError("ninput does not match the shape of the provided inputs.")
        if datatype != "image" and datatype != "spectro":
            raise ValueError("Specified data type not supported. Supported types are image \
            and spectro. For vectors of PC coefficients, input the \
            PCA object used to create the PC projections.")

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
        if imshape is None:
            linput = np.sqrt(self.ninput)
            self.imshape = (int(linput), int(linput))
            if linput != self.imshape[0]:
                raise ValueError("Input size not a perfect square. Please provide image shape.")
        else:
            self.imshape = imshape
                
        # initialize network parameters
        # Q are feedfoward weights (i.e. from input units to output units)
        # W are horizontal conections (among 'output' units)
        # theta are thresholds for the LIF neurons
        self.Q = self.rand_dict()
        self.W = np.zeros((self.nunits, self.nunits))
        self.theta = theta0*np.ones(self.nunits)        
        
        # initialize average activity stats
        self.meanact_ave = self.p
        self.corrmatrix_ave = self.p**2
        
        #initialize history of objective function
        self.objhistory = np.array([])
        self.errorhistory = np.array([])
  
    def rand_dict(self):
        Q = np.random.randn(self.nunits, self.ninput)
        normmatrix = np.diag(1/np.sqrt(np.sum(Q*Q,1))) 
        return np.dot(normmatrix,Q) # normalize initial feedforward weight vectors
      
    def infer(self, X, do_inference_plot = False):
        """
        Simulate LIF neurons to get spike counts. Optionally plot mean square reconstruction error vs time.
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
        
        if do_inference_plot:
            errors = np.zeros(self.niter)
        
        for t in range(self.niter):
            # DE for internal variables
            u = (1.-eta)*u + eta*(B - np.dot(self.W,y))
            
            # external variables should spike when internal variables cross threshholds
            y = np.array([u[:,ind] >= self.theta for ind in range(self.batch_size)])
            y = y.T

            # add spikes to counts
            acts = acts + y
            
            if do_inference_plot:
                errors[t] = np.mean(self.compute_errors(acts, X))
            
            # reset the internal variables of the spiking units
            u = u*(1-y)
        
        if do_inference_plot:
            plt.figure(3)
            plt.clf()
            plt.plot(errors)
        
        return acts  
        
    def showrfs(self, cmap = None):
        """Plot receptive fields, tiled in one big image. Default color map is
        gray for images and jet for spectrograms."""
        if self.pca is not None:
            ffweights = self.pca.inverse_transform(self.Q)
        else:
            ffweights = self.Q
            
        # length and height of each individual RF        
        length, height = self.imshape
        assert length*height == ffweights.shape[1]
        buf = 1 # buffer pixel(s) between RFs
        
        M = self.nunits
        
        # n and m are number of rows and columns of RFs in the array
        if np.floor(np.sqrt(M))**2 != M:
            n = int(np.ceil(np.sqrt(M/2.)))
            m = int(np.ceil(M/n))
        else:
            # M is a perfect square
            m = int(np.sqrt(M))
            n = m
            
        array = 0.5*np.ones((buf+n*(height+buf), buf+m*(length+buf)))
        k = 0
        
        # TODO: make this less ugly
        # Right now it loops over every pixel in the array of STRFs.
        for i in range(n):
            for j in range(m):
                if k < M:
                    normfactor = np.max(np.abs(ffweights[k,:]))
                    for li in range(height):
                        for lj in range(length):
                            array[buf+(i)*(height+buf)+li, buf+(j)*(length+buf)+lj] =  \
                            ffweights[k,lj+length*li]/normfactor
                k = k+1
        
        cmap = cmap or ('jet' if self.datatype == 'spectro' else 'gray')
        arrayplot = plt.imshow(array,interpolation='nearest', cmap=cmap, aspect='auto')
        plt.colorbar()
        return arrayplot
    
    def show_network(self):
        """
        Plot current values of weights, thresholds, and time-averaged firing
        correlations.
        """
        
        plt.subplot(2,2,1)
        plt.imshow(self.W, cmap = "gray", interpolation="nearest",aspect='auto')
        plt.colorbar()
        plt.title("Inhibitory weights")
        
        plt.subplot(2,2,2)
        C = self.corrmatrix_ave - \
            np.dot(self.meanact_ave, np.transpose(self.meanact_ave))
        plt.imshow(C, cmap = "gray", interpolation="nearest",aspect = 'auto')
        plt.colorbar()
        plt.title("Moving time-averaged correlation")
        
        plt.subplot(2,2,3)
        plt.plot(np.concatenate([self.objhistory[:,None]/np.mean(self.objhistory),
                                 self.errorhistory[:,None]/np.mean(self.errorhistory)], 1))
        plt.title("History of objective function and error")
        #self.showrfs()
        #plt.title("Feedforward weights")
        
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
                row = self.buffer + int(np.ceil((self.imsize- 
                self.lpatch-2*self.buffer)*np.random.rand()))
                col = self.buffer + int(np.ceil((self.imsize- 
                self.lpatch-2*self.buffer)*np.random.rand()))
                animage = self.images[row:row+self.lpatch,
                                      col:col+self.lpatch,
                                      int(np.floor(self.nimages*np.random.rand()))]                     
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
            whichimage = int(np.floor(self.nimages*np.random.rand()))
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
            whichvec = int(np.floor(self.nimages*np.random.rand()))
            avec = self.images[:, whichvec]
            avec = avec - np.mean(avec)
            avec = avec/np.std(avec)
            X[:,i] = avec
        return X        
    
    def learn(self, X, acts, corrmatrix):
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
            
    
    def run(self, ntrials = 25000):
        """
        Run SAILnet for ntrials: for each trial, create a random set of image
        patches, present each to the network, and update the network weights
        after each set of batch_size presentations.
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
            acts = self.infer(X)
            
            # compute statistics for this batch
            meanact = np.mean(acts,1)
            corrmatrix = np.dot(acts, np.transpose(acts))/self.batch_size 
            
            # update weights and thresholds according to learning rules
            self.learn(X, acts, corrmatrix)
            
            # compute moving averages of meanact and corrmatrix
            self.meanact_ave = (1 - self.eta_ave)*self.meanact_ave + \
                self.eta_ave*meanact
            self.corrmatrix_ave = (1 - self.eta_ave)*self.corrmatrix_ave + \
                self.eta_ave*corrmatrix
                
            # save statistics every 50 trials
            if t % 50 == 0:
                # Compute and save current value of objective function
                self.objhistory = np.append(self.objhistory, 
                                            self.compute_objective(acts,X))
                self.errorhistory = np.append(self.errorhistory, 
                                              np.sum(self.compute_errors(acts, X)))
                
                print("Trial number: " + str(t))
                if t % 5000 == 0:
                    # save progress
                    print("Saving progress...")
                    self.save_params()
                    print("Done. Continuing to run...")
                    
        # Save final parameter values            
        self.save_params()              
     
    def visualize(self):
        """Display visualizations of network parameters."""
        plt.figure(1)
        plt.clf()
        self.show_network()
        plt.figure(2)
        plt.clf()
        self.showrfs()
     
    def generate_model(self, acts):
        """Reconstruct inputs using linear generative model."""
        return np.dot(self.Q.T,acts)
        
    def compute_errors(self, acts, X):
        """Given a batch of data and activities, compute the squared error between
        the generative model and the original data. Returns a vector of squared errors."""
        diffs = X - self.generate_model(acts)
        return np.sum(diffs**2,axis=0)
        
    def compute_objective(self, acts, X):
        """Compute value of objective function/Lagrangian averaged over batch."""
        errorterm = np.sum(self.compute_errors(acts, X))
        thetarep = np.repeat(self.theta[:,np.newaxis], self.batch_size,axis=1)
        rateterm = -np.sum(thetarep*(acts - self.p))
        corrWmatrix = np.dot(np.dot(np.transpose(acts), self.W),acts)
        corrterm = -(1/self.batch_size)*np.trace(corrWmatrix) + np.sum(self.W)*self.p**2
        return (errorterm*self.beta/2 + rateterm*self.gamma + corrterm*self.alpha)/self.batch_size
        
    
    def save_params(self, filename=None):
        """
        Save parameters to a pickle file, to be picked up later. By default
        we save to the file name stored with the SAILnet instance, but a different
        file can be passed in as the string filename. This filename is then saved.
        """
        if filename is None:
            filename = self.picklefile
        with open(filename,'wb') as f:
            pickle.dump([self.Q, self.W, self.theta, 
                         self.meanact_ave, self.corrmatrix_ave], f)
        self.picklefile = filename

    def load_params(self, filename = None):
        """Load parameters (e.g., weights) from a previous run from pickle file.
        This pickle file is then associated with this instance of SAILnet."""
        if filename is None:
            filename = self.picklefile
        with open(filename, 'rb') as f:
            self.Q, self.W, self.theta, self.meanact_ave, self.corrmatrix_ave = \
            pickle.load(f)               
        self.picklefile = filename
            
    def adjust_rates(self, factor):
        """Multiply all the learning rates (alpha, beta, gamma) by the given factor."""
        self.alpha = factor*self.alpha
        self.beta = factor*self.beta
        self.gamma = factor*self.gamma
        self.objhistory = factor*self.objhistory
        
            
            