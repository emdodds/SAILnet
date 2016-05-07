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
import StimSet
import plotting

class SAILnet(DictLearner):
    """
    Runs SAILnet: Sparse And Independent Local Network
    Currently supports images and spectrograms.
    """
    
    def __init__(self,
                 data = None,
                 datatype = "image",
                 stimshape = None,
                 batch_size = 100,
                 niter = 50,
                 delay = 0,
                 buffer = 20,
                 ninput = 256,
                 nunits = 256,
                 p = 0.05,
                 alpha = 1.,
                 beta = 0.01,
                 gamma = 0.1,
                 theta0 = 0.5,
                 infrate = 0.1,
                 moving_avg_rate = 0.001,
                 picklefile = 'SAILnetparams.pickle',
                 pca = None):
        """
        Create SAILnet object with given parameters. 
        Defaults are as used in Zylberberg et al.
        
        Args:
        data:               numpy array of data for analysis
        datatype:           type of data in imagefile. Images and spectrograms
                            are supported. For PC projections representing
                            spectrograms, use spectro and input the relevant 
                            PCA object. Images are assumed to be squares.
        stimshape:            Shape of images/spectrograms. Square by default.
        batch_size:         number of image presentations between each learning step
        niter:              number of time steps in calculation of activities
                            for one image presentation
        delay:              spikes before this time step don't count towards reconstruction and learning
        buffer:             buffer on image edges
        ninput:             number of input units: for images, number of pixels
        nunits:             number of output units
        p:                  target firing rate
        alpha:              learning rate for inhibitory weights
        beta:               learning rate for feedforward weights
        gamma:              learning rate for thresholds
        theta0:             initial value of thresholds
        infrate:            rate parameter for inference with LIF circuit
        moving_avg_rate:    rate parameter for computing moving averages to get activity stats
        picklefile:         File in which to save pickled parameters.
        pca:                PCA object used to create vector inputs.
                            Used here to reconstruct spectrograms for display.
                            
        Raises:
        ValueError when datatype is not one of the supported options.
        """
        # If no input data passed in, use IMAGES.mat  
        if data is None:
            data = scipy.io.loadmat("../Data/IMAGES.mat")["IMAGES"]
        
        self.datatype = datatype
        
        if self.datatype == "spectro" and data.shape[1] != ninput:
            if data.shape[0] == ninput:
                # If the array is passed in with the indices swapped, transpose it
                data = np.transpose(data)
            else:
                raise ValueError("ninput does not match the shape of the provided inputs.")
            

        # Store instance variables
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.infrate = infrate
        self.moving_avg_rate = moving_avg_rate
        self.p = p        
        self.batch_size = batch_size
        self.niter = niter
        self.delay = delay
        self.nunits = nunits # M in original MATLAB code
        self.picklefile = picklefile
        self.pca = pca
        self.plotter = plotting.Plotter(self)
        
        # size of patches
        self.ninput = ninput # N in original MATLAB code
        if stimshape is None:
            linput = np.sqrt(self.ninput)
            stimshape = (int(linput), int(linput))
            if linput != stimshape[0]:
                raise ValueError("Input size not a perfect square. Please provide image shape.")
            
        if datatype == "spectro":
            self.stims = StimSet.PCvecSet(data, stimshape, self.pca, self.batch_size)
        elif datatype == "image":
            self.stims = StimSet.ImageSet(data, batch_size=batch_size, buffer=buffer, stimshape=stimshape)
        else:
            raise ValueError("Specified data type not supported. Supported types are image \
            and spectro. For vectors of PC coefficients, input the \
            PCA object used to create the PC projections.")
                
        self.initialize(theta0)
        
        
    def initialize(self, theta0=0.5):
        """Initialize or reset weights, averages, histories."""
        # initialize network parameters
        # Q are feedfoward weights (i.e. from input units to output units)
        # W are horizontal conections (among 'output' units)
        # theta are thresholds for the LIF neurons
        self.Q = self.rand_dict()
        self.W = np.zeros((self.nunits, self.nunits))
        self.theta = theta0*np.ones(self.nunits)        
        
        # initialize average activity stats
        self.corrmatrix_ave = self.p**2
        self.L0acts = np.zeros(self.nunits)
        self.L1acts = np.zeros(self.nunits)
        
        #initialize history of objective function
        self.objhistory = np.array([])
        self.errorhistory = np.array([])
      
    def infer(self, X, infplot = False, savestr = None):
        """
        Simulate LIF neurons to get spike counts. Optionally plot mean square reconstruction error vs time.
        X:        input array
        Q:        feedforward weights
        W:        horizontal weights
        theta:    thresholds
        y:        outputs
        """
        
        nstim = X.shape[-1]        
        
        # projections of stimuli onto feedforward weights
        B = np.dot(self.Q,X)

        # initialize values. Note that I've renamed some variables compared to 
        # Zylberberg's code. My variable names more closely follow the paper instead.
        u = np.zeros((self.nunits, nstim)) # internal unit variables
        y = np.zeros((self.nunits, nstim)) # external unit variables
        acts = np.zeros((self.nunits, nstim)) # counts of total firings
        
        if infplot:
            errors = np.zeros(self.niter)
            yhist = np.zeros((self.niter))
        
        for t in range(self.niter):
            # DE for internal variables
            u = (1.-self.infrate)*u + self.infrate*(B - 2*self.W.dot(y))
            
            # external variables should spike when internal variables cross threshholds
            y = np.array([u[:,ind] >= self.theta for ind in range(nstim)])
            y = y.T

            # add spikes to counts (if using delay, don't count spikes before delay time)
            if t>=self.delay:
                acts = acts + y
            
            if infplot:
                errors[t] = np.mean(self.compute_errors(acts, X))
                yhist[t] = np.mean(y)
            
            # reset the internal variables of the spiking units
            u = u*(1-y)
        
        if infplot:
            self.plotter.inference_plots(errors, yhist, savestr=savestr)
        
        return acts
    
    def pairwisedot(self, acts=None):
        pairdots = []
        if acts is None:
            for i in range(self.nunits):
                for j in range(i,self.nunits):
                    pairdots.append(self.Q[i].dot(self.Q[j]))
            return pairdots
        # if acts is provided, only return pairwise dot products for coactive units in a batch
        for it in range(self.batch_size):
            itacts = acts[:,it]
            itdots = []
            for unit in range(self.nunits):
                if itacts[unit] >0:
                    for other in range(unit,self.nunits):
                        if itacts[other] > 0:
                            pairdots.append(self.Q[unit].dot(self.Q[other]))
        return pairdots
                               
    
    def learn(self, X, acts, corrmatrix):
        """Use learning rules to update network parameters."""
        
        # update feedforward weights with Oja's rule
        sumsquareacts = np.sum(acts*acts,1) # square, then sum over images
        dQ = acts.dot(X.T) - np.diag(sumsquareacts).dot(self.Q)
        self.Q = self.Q + self.beta*dQ/self.batch_size        
        
        #acts = acts > 0 # This and below makes the W and theta rules care about L0 activity # TODO: REMOVE REMOVE REMOVE REMOVE REMOVE
        #corrmatrix = np.dot(acts, np.transpose(acts))/self.batch_size

        # update lateral weights with Foldiak's rule 
        # (inhibition for decorrelation)
        dW = self.alpha*(corrmatrix - self.p**2)
        self.W = self.W + dW
        self.W = self.W - np.diag(np.diag(self.W)) # zero diagonal entries
        self.W[self.W < 0] = 0 # force weights to be inhibitory
        
        # update thresholds with Foldiak's rule: keep firing rates near target
        dtheta = self.gamma*(np.sum(acts,1)/self.batch_size - self.p)
        self.theta = self.theta + dtheta
            
    
    def run(self, ntrials = 25000, rate_decay=1):
        """
        Run SAILnet for ntrials: for each trial, create a random set of image
        patches, present each to the network, and update the network weights
        after each set of batch_size presentations.
        The learning rates area all multiplied by the factor rate_decay after each trial.
        """
        for t in range(ntrials):
            # make data array X from random pieces of total data
            X = self.stims.rand_stim()
            
            # compute activities for this data array
            acts = self.infer(X)
            
            # compute statistics for this batch
            meanact = np.mean(acts,1)
            corrmatrix = np.dot(acts, acts.T)/self.batch_size 
            
            # update weights and thresholds according to learning rules
            self.learn(X, acts, corrmatrix)
            
            # compute moving averages of activities and corrmatrix
            self.corrmatrix_ave = (1 - self.moving_avg_rate)*self.corrmatrix_ave + self.moving_avg_rate*corrmatrix
            self.L1acts = (1-self.moving_avg_rate)*self.L1acts + self.moving_avg_rate*meanact
            L0means = np.mean(acts != 0,axis=1).astype('float32')
            self.L0acts = (1-self.moving_avg_rate)*self.L0acts + self.moving_avg_rate*L0means
                
            # save statistics every 50 trials
            if t % 50 == 0:
                # Compute and save current value of objective function
                self.objhistory = np.append(self.objhistory, 
                                            self.compute_objective(acts,X))
                self.errorhistory = np.append(self.errorhistory, 
                                              np.mean(self.compute_errors(acts, X)))
                
                print("Trial number: " + str(t))
                if t % 5000 == 0:
                    # save progress
                    print("Saving progress...")
                    self.save_params()
                    print("Done. Continuing to run...")
            
            # adjust rates
            self.adjust_rates(rate_decay)
            
        # Save final parameter values            
        self.save_params()              
     
    def visualize(self):
        """Display visualizations of network parameters."""
        self.plotter.visualize()
        
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
        histories = (self.L0acts,self.L1acts, self.corrmatrix_ave, self.errorhistory, self.objhistory)
        rates = (self.alpha, self.beta, self.gamma)
        with open(filename,'wb') as f:
            pickle.dump([self.Q, self.W, self.theta, rates, histories], f)
        self.picklefile = filename

    def load_params(self, filename = None):
        """Load parameters (e.g., weights) from a previous run from pickle file.
        This pickle file is then associated with this instance of SAILnet."""
        if filename is None:
            filename = self.picklefile
        with open(filename, 'rb') as f:
            self.Q, self.W, self.theta, rates, histories = pickle.load(f)  
        try:
            try:
                self.L0acts, self.L1acts, self.corrmatrix_ave, self.errorhistory, self.objhistory = histories
                assert len(self.L1acts) < 2
            except AssertionError:
                self.L1acts, self.corrmatrix_ave, self.errorhistory, self.objhistory, usage = histories
                self.L0acts = usage
        except ValueError:
            self.L1acts, self.corrmatrix_ave, self.errorhistory, self.objhistory = histories
        self.alpha, self.beta, self.gamma = rates
        self.picklefile = filename
            
    def adjust_rates(self, factor):
        """Multiply all the learning rates (alpha, beta, gamma) by the given factor."""
        self.alpha = factor*self.alpha
        self.beta = factor*self.beta
        self.gamma = factor*self.gamma
        self.objhistory = factor*self.objhistory
        
    def sort_dict(self, batch_size=None, allstims=False, plot=True, savestr=None):
        """Sorts the feedforward RFs in order by their usage on a batch.
        By default the batch used for this computation is 10x the usual one,
        since otherwise many dictionary elements tend to have zero activity."""
        batch_size = batch_size or 10*self.batch_size
        if allstims:
            testX = self.stims.data.T
        else:
            testX = self.stims.rand_stim(batch_size)
        #meanacts = np.mean(self.infer(testX),axis=1)   
        usage = np.mean(self.infer(testX) != 0,axis=1)
        sorter = np.argsort(usage)
        self.sort(usage, sorter, plot, savestr)
        self.L0acts = usage[sorter]
        return self.L0acts
        
    def fast_sort(self, plot = True, savestr=None):
        """Sort the feedforward RFs in order by their moving time-averaged activities."""
        sorter = np.argsort(self.L0acts)
        self.sort(self.L0acts, sorter, plot, savestr)
            
    def sort(self, usage, sorter, plot=False, savestr=None):
        self.Q = self.Q[sorter]
        self.W = self.W[sorter]
        self.W = self.W.T[sorter].T
        self.theta = self.theta[sorter]
        self.L0acts = self.L0acts[sorter]
        self.L1acts = self.L1acts[sorter]
        self.corrmatrix_ave = self.corrmatrix_ave[sorter]
        self.corrmatrix_ave = self.corrmatrix_ave.T[sorter].T
        if plot:
            plt.figure()
            plt.plot(usage[sorter])
            plt.title('Mean L0 activity for each unit, sorted')
            if savestr is not None:
                plt.savefig(savestr, bbox_inches='tight')
                
    def test_inference(self):
        X = self.stims.rand_stim()
        s = self.infer(X, infplot=True)
        print("Final SNR: " + str(self.snr(X,s)))
        return s