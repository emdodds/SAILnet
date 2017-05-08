# -*- coding: utf-8 -*-
"""
Created on Sat May 14 19:43:16 2016

@author: Eric
"""

import SAILnet
import numpy as np


class VarTimeSAILnet(SAILnet.SAILnet):
    """SAILnet with a rate code (as opposed to a count code)
    and variable inference time."""
    def __init__(self, inftime=5, infrate=0.1, gain_rate=0.001, **kwargs):
        """Args:
        inftime: inference time in time-constants of the circuit
        infrate: integration step size, also duration of a spike
        all other arguments same as SAILnet, use keywords
        """
        self.inftime = inftime
        self.gain = 1.0
        self.gain_rate = gain_rate
        niter = int(np.ceil(self.inftime/infrate))
        super().__init__(niter=niter, **kwargs)

    def compute_gain(self, X, acts):
        recon = self.generate_model(acts)
        denom = np.mean(recon**2)
        if denom == 0:
            gain = self.gain
        else:
            gain = np.mean(X**2) / np.mean(recon**2)
        # clip gain if it gets too far from 1
        gain = min(gain, 10.0)
        gain = max(gain, 0.1)
        return gain

    def learn(self, X, acts, corrmatrix):
        """Use learning rules to update network parameters."""

        # update feedforward weights with Oja's rule
        sumsquareacts = np.sum(acts*acts, 1)  # square, then sum over images
        dQ = acts.dot(X.T) - np.diag(sumsquareacts).dot(self.Q)
        self.Q = self.Q + self.beta*dQ/self.batch_size

        # update lateral weights with Foldiak's rule
        # (inhibition for decorrelation)
        dW = self.alpha*(corrmatrix - self.p**2)
        self.W = self.W + dW
        self.W = self.W - np.diag(np.diag(self.W))  # zero diagonal entries
        self.W[self.W < 0] = 0  # force weights to be inhibitory

        # update thresholds with Foldiak's rule: keep firing rates near target
        dtheta = self.gamma*(np.sum(acts, 1)/self.batch_size - self.p)
        self.theta = self.theta + dtheta

        self.gain = self.gain*(self.compute_gain(X, acts)/self.gain)**self.gain_rate

    def infer(self, X, infplot=False, savestr=None):
        """
        Simulate LIF neurons to get spike counts.
        Optionally plot mean square reconstruction error vs time.
        X:        input array
        Q:        feedforward weights
        W:        horizontal weights
        theta:    thresholds
        y:        outputs
        """
        nstim = X.shape[-1]

        # projections of stimuli onto feedforward weights
        B = np.dot(self.Q, X)

        # initialize values. Note that I've renamed some variables compared to
        # Zylberberg's code. My variable names more closely follow the paper
        u = np.zeros((self.nunits, nstim))  # internal unit variables
        y = np.zeros((self.nunits, nstim))  # external unit variables
        acts = np.zeros((self.nunits, nstim))  # counts of total firings

        if infplot:
            errors = np.zeros(self.niter)
            yhist = np.zeros((self.niter))

        for t in range(self.niter):
            # DE for internal variables
            u = (1.-self.infrate)*u + self.infrate*(B - 2*self.W.dot(y))

            # external variables spike when internal variables cross thresholds
            y = np.array([u[:, ind] >= self.theta for ind in range(nstim)])
            y = y.T

            acts = acts + y

            if infplot:
                recon_t = self.gain*acts/((t+1)*self.infrate)
                errors[t] = np.mean(self.compute_errors(recon_t, X))
                yhist[t] = np.mean(y)

            # reset the internal variables of the spiking units
            u = u*(1-y)

        if infplot:
            self.plotter.inference_plots(errors, yhist, savestr=savestr)

        return self.gain*acts/self.inftime

    def get_param_list(self):
        params = super().get_param_list()
        params['gain'] = self.gain
        return params


class FreeGainSAILnet(VarTimeSAILnet):
    """Like VarTimeSAILnet, but the homeostatic rules control the spike
    counts unscaled by the gain."""
    def learn(self, X, acts, corrmatrix):
        """Use learning rules to update network parameters."""

        # update feedforward weights with Oja's rule
        sumsquareacts = np.sum(acts*acts, 1)  # square, then sum over images
        dQ = acts.dot(X.T) - np.diag(sumsquareacts).dot(self.Q)
        self.Q = self.Q + self.beta*dQ/self.batch_size

        # homeostatic rules pay attention to spike rate, regardless of gain
        acts = acts/self.gain
        # update lateral weights with Foldiak's rule
        # (inhibition for decorrelation)
        dW = self.alpha*(corrmatrix - self.p**2)
        self.W = self.W + dW
        self.W = self.W - np.diag(np.diag(self.W))  # zero diagonal entries
        self.W[self.W < 0] = 0  # force weights to be inhibitory

        # update thresholds with Foldiak's rule: keep firing rates near target
        dtheta = self.gamma*(np.sum(acts, 1)/self.batch_size - self.p)
        self.theta = self.theta + dtheta

        self.gain = self.gain*(self.compute_gain(X, acts)/self.gain)**self.gain_rate


class NLnet(VarTimeSAILnet):
    """Uses nonlocal learning rule. Inference is VarTimeSAILnet inference."""
    def learn(self, X, acts, corrmatrix):
        """Use learning rules to update network parameters."""
                
        R = X.T - np.dot(acts.T, self.Q)
        self.Q = self.Q + self.beta*np.dot(acts,R)/X.shape[1]
        
        # update lateral weights with Foldiak's rule 
        # (inhibition for decorrelation)
        dW = self.alpha*(corrmatrix - self.p**2)
        self.W = self.W + dW
        self.W = self.W - np.diag(np.diag(self.W)) # zero diagonal entries
        self.W[self.W < 0] = 0 # force weights to be inhibitory
        
        # update thresholds with Foldiak's rule: keep firing rates near target
        dtheta = self.gamma*(np.sum(acts,1)/self.batch_size - self.p)
        self.theta = self.theta + dtheta
        
class LCAILnet(SAILnet.SAILnet):
    """
    A SAILnet-LCA hybrid. Inference is like LCA except activities are only positive, 
    the inhibitory connection strengths are learned as in SAILnet, and the 
    thresholds (lambda) are learned as in SAILnet. The dictionary learning rule
    is gradient descent, not the approximate SAILnet rule.
    """
    def infer(self, X, infplot=False, savestr=None):
        ndict = self.Q.shape[0]
        thresh = self.theta

        nstim = X.shape[-1]
        u = np.zeros((nstim, ndict))
        s = np.zeros_like(u)
        ci = np.zeros_like(u)

        # dot-product inhibition replaced by learned inhibition
        c = self.W

        # b[i,j] is overlap of stimulus i with dictionary element j
        b = (self.Q.dot(X)).T

        if infplot:
            errors = np.zeros(self.niter)
            yhist = np.zeros((self.niter))

        for kk in range(self.niter):
            # ci is the competition term in the dynamical equation
            ci[:] = s.dot(c)
            u[:] = self.infrate*(b-ci) + (1.-self.infrate)*u
            if np.max(np.isnan(u)):
                raise ValueError("Internal variable blew up at iteration " + str(kk))
            #if self.softthresh:
             #   s[:] = np.sign(u)*np.maximum(0.,np.absolute(u)-thresh[:,np.newaxis]) 
            else:
                s[:] = u
                #s[np.absolute(s) < thresh] = 0
                s[s<thresh] = 0

            if infplot:
                errors[kk] = np.mean((X.T - s.dot(self.Q))**2)
                yhist[kk] = np.mean(s)

        if infplot:
            self.plotter.inference_plots(errors, yhist, savestr=savestr)
            
        return s.T
    
    def learn(self, X, acts, corrmatrix):
        """Use learning rules to update network parameters."""

        # update feedforward weights with Oja's rule (Nonlocal)
        dQ = acts.dot(X.T - acts.T.dot(self.Q))
        self.Q = self.Q + self.beta*dQ/self.batch_size       

        #acts = acts > 0 # This and below makes the W and theta rules care about L0 activity # TODO: REMOVE 
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
        self.theta[self.theta<0] = 0
