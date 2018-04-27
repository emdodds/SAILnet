# -*- coding: utf-8 -*-
"""
Created on Sat May 14 19:43:16 2016

@author: Eric
"""

from SAILnet import SAILnet
import numpy as np


class VarTimeSAILnet(SAILnet.SAILnet):
    """SAILnet with a rate code (as opposed to a count code)
    and variable inference time."""
    def __init__(self, inftime=5, infrate=0.1,
                 gain_rate=0, gain=1.0, **kwargs):
        """Args:
        inftime: inference time in time-constants of the circuit
        infrate: integration step size, also duration of a spike
        gain: controls the magnitude of a spike
        gain_rate: rate at which to update gain to optimize reconstrution
        all other arguments same as SAILnet, use keywords
        """
        self.inftime = inftime
        self.gain = gain
        self.gain_rate = gain_rate
        niter = int(np.ceil(self.inftime/infrate))
        super().__init__(niter=niter, **kwargs)

    def compute_gain(self, X, acts):
        """Compute gain that would match response variance to input
        variance, but trimmed if too far from 1."""
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

        if self.gain_rate > 0:
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

        # update gain
        self.gain = self.gain*(self.compute_gain(X, acts)/self.gain)**self.gain_rate

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
    def __init__(self, *args, softthresh=False, nonneg=True, **kwargs):
        self.softthresh = softthresh
        self.nonneg = nonneg
        SAILnet.SAILnet.__init__(self, *args, **kwargs)

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
            if self.softthresh:
                if self.nonneg:
                    s[:] = np.maximum(0., u-thresh[np.newaxis, :])
                else:
                    s[:] = np.sign(u)*np.maximum(0.,
                        np.absolute(u)-thresh[np.newaxis, :])
            else:
                s[:] = u
                if self.nonneg:
                    s[s < thresh] = 0
                else:
                    s[np.absolute(s) < thresh] = 0

            if infplot:
                errors[kk] = np.mean((X.T - s.dot(self.Q))**2)
                yhist[kk] = np.mean(np.abs(s))

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


class LocalDictRuleLCA(LCAILnet):
    """SAILnet with W fixed at Gram matrix of Phi and LCA-like inference.
    Equivalently, LCA with local Phi learning rule.
    Exception: Q is normalized."""
    def learn(self, X, acts, corrmatrix):
        # update feedforward weights with Oja's rule
        sumsquareacts = np.sum(acts*acts, 1)  # square, then sum over images
        dQ = acts.dot(X.T) - np.diag(sumsquareacts).dot(self.Q)
        self.Q = self.Q + self.beta*dQ/self.batch_size
        normmatrix = np.diag(1./np.sqrt(np.sum(self.Q*self.Q, 1)))
        self.Q = normmatrix.dot(self.Q)

        self.W = self.Q.dot(self.Q.T)

        # update thresholds with Foldiak's rule: keep firing rates near target
        dtheta = self.gamma*(np.sum(acts, 1)/self.batch_size - self.p)
        self.theta = self.theta + dtheta


class LCALocalLearner(LCAILnet):
    """LCA-like inference with SAILnet structure and learning rules.
    +/- symmetric by default. Lateral weights not constrained to be
    inhibitory."""
    def __init__(self, *args, softthresh=True, nonneg=False, **kwargs):
        LCAILnet.__init__(self, *args, softthresh=softthresh,
                          nonneg=nonneg, **kwargs)

    def learn(self, X, acts, corrmatrix):
        """Use learning rules to update network parameters."""

        # update feedforward weights with Oja's rule
        sumsquareacts = np.sum(acts*acts, 1)  # square, then sum over images
        dQ = acts.dot(X.T) - np.diag(sumsquareacts).dot(self.Q)
        self.Q = self.Q + self.beta*dQ/self.batch_size

        # update lateral weights with Foldiak's rule
        # (inhibition for decorrelation)
        corr = corrmatrix
        if self.nonneg:
            corr -= self.p**2
        dW = self.alpha*corr
        self.W = self.W + dW
        self.W = self.W - np.diag(np.diag(self.W))  # zero diagonal entries
        # self.W[self.W < 0] = 0  # force weights to be inhibitory

        # individual synapses should not be too strong. clip them.
        # TODO: think of less hacky way to handle explosion problem
        toobigW = np.abs(self.W) > 1
        self.W[toobigW] = np.sign(self.W[toobigW])
        toobigQ = np.abs(self.Q) > 1
        self.Q[toobigQ] = np.sign(self.Q[toobigQ])

        # update thresholds with Foldiak's rule: keep firing rates near target
        dtheta = self.gamma*(np.sum(np.abs(acts), 1)/self.batch_size - self.p)
        self.theta = self.theta + dtheta

    def compute_objective(self, acts, X):
        """Compute value of objective/Lagrangian averaged over batch."""
        errorterm = np.mean(self.compute_errors(acts, X))
        rateterm = np.mean((np.abs(acts)-self.p)*self.theta[:, np.newaxis])
        corrWmatrix = acts.T.dot(self.W).dot(acts)
        corrterm = (1/acts.shape[1]**2)*np.trace(corrWmatrix)
        return (errorterm*self.beta/2 + rateterm*self.gamma +
                corrterm*self.alpha)

    def set_dot_inhib(self):
        """Sets each lateral weight to the dot product of the corresponding
        units' feedforward weights."""
        self.W = self.Q.dot(self.Q.T)
        self.W = self.W - np.diag(np.diag(self.W))  # zero diagonal entries
        # self.W[self.W < 0] = 0  # force weights to be inhibitory

    def initialize(self, theta0=0.5):
        """Initialize or reset weights, averages, histories."""
        # Q are feedfoward weights (i.e. from input units to output units)
        # W are horizontal conections (among 'output' units)
        # theta are thresholds for the LIF neurons
        self.Q = self.rand_dict()
        self.W = np.zeros((self.nunits, self.nunits))
        self.theta = theta0*np.ones(self.nunits)

        # initialize average activity stats
        self.initialize_stats()
        self.corrmatrix_ave = 0
        self.objhistory = np.array([])


class MirrorSAIL(SAILnet.SAILnet):
    """SAILnet but each unit has a partner with the opposite feedforward weights."""
    def initialize(self, theta0=0.5):
        """Initialize or reset weights, averages, histories."""
        # initialize network parameters
        # Q are feedfoward weights (i.e. from input units to output units)
        # W are horizontal conections (among 'output' units)
        # theta are thresholds for the LIF neurons
        self.Q = self.rand_dict()
        self.W = np.zeros((2*self.nunits, 2*self.nunits))
        self.theta = theta0*np.ones(self.nunits)
        self.thetam = theta0*np.ones(self.nunits)

        # initialize average activity stats
        self.nunits = 2*self.nunits
        self.initialize_stats()
        self.nunits = int(self.nunits/2)
        self.corrmatrix_ave = self.p**2
        self.objhistory = np.array([])
    
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
        ndict = self.Q.shape[0]
        
        # projections of stimuli onto feedforward weights
        B = np.dot(self.Q,X)

        # initialize values. Note that I've renamed some variables compared to 
        # Zylberberg's code. My variable names more closely follow the paper instead.
        u = np.zeros((self.nunits, nstim)) # internal unit variables
        um = np.zeros_like(u)
        y = np.zeros_like(u) # external unit variables
        ym = np.zeros_like(u)
        acts = np.zeros_like(u) # counts of total firings
        actsm = np.zeros_like(u)
        
        if infplot:
            errors = np.zeros(self.niter)
            yhist = np.zeros((self.niter))
        
        for t in range(self.niter):
            # DE for internal variables
            u = (1.-self.infrate)*u + self.infrate*(B - 2*self.W[:ndict,:ndict].dot(y) - 2*self.W[:ndict,ndict:].dot(ym))
            um = (1.-self.infrate)*um + self.infrate*(-B -2*self.W[ndict:,:ndict].dot(y) - 2*self.W[ndict:, ndict:].dot(ym))
            
            # external variables should spike when internal variables cross threshholds
            y = np.array([u[:,ind] >= self.theta for ind in range(nstim)])
            y = y.T
            ym = np.array([um[:,ind] >= self.thetam for ind in range(nstim)])
            ym = ym.T

            acts = acts + y
            actsm = actsm + ym
            
            if infplot:
                errors[t] = np.mean(self.compute_errors(acts,actsm, X))
                yhist[t] = (np.mean(y) + np.mean(ym))/2
            
            # reset the internal variables of the spiking units
            u = u*(1-y)
            um = um*(1-ym)
        
        if infplot:
            self.plotter.inference_plots(errors, yhist, savestr=savestr)
        
        return acts, actsm
    
    def learn(self, X, acts,actsm, corrmatrix):
        """Use learning rules to update network parameters."""
        
        # update feedforward weights with Oja's rule
        sumsquareacts = np.sum(acts*acts,1) # square, then sum over images
        sumsquareactsm = np.sum(actsm**2,1)
        crossterm = np.sum(acts*actsm,1)
        dQ = (acts-actsm).dot(X.T) - np.diag(sumsquareacts+sumsquareactsm-2*crossterm).dot(self.Q)
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
        dthetam = self.gamma*(np.sum(actsm,1)/self.batch_size - self.p)
        self.thetam = self.thetam + dthetam
        
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
            acts, actsm = self.infer(X)
            allacts = np.concatenate([acts, actsm])

            # compute statistics for this batch
            errors = np.mean(self.compute_errors(acts, actsm, X))
            corrmatrix = self.store_statistics(allacts, errors)
            self.objhistory = np.append(self.objhistory,
                                        self.compute_objective(acts, actsm, X))

            # update weights and thresholds according to learning rules
            self.learn(X, acts, actsm, corrmatrix)

            # save statistics every 50 trials
            if t % 50 == 0:
                print("Trial number: " + str(t))
                if t % 5000 == 0:
                    # save progress
                    print("Saving progress...")
                    self.save()
                    print("Done. Continuing to run...")

            self.adjust_rates(rate_decay)

        self.save()

    def generate_model(self, acts, actsm):
        """Reconstruct inputs using linear generative model."""
        return np.dot(self.Q.T, acts) - np.dot(self.Q.T, actsm)

    def compute_errors(self, acts, actsm, X):
        """Given a batch of data and activities, compute the squared error between
        the generative model and the original data. Returns vector of mean squared errors."""
        diffs = X - self.generate_model(acts, actsm)
        return np.mean(diffs**2, axis=0)

    def compute_objective(self, acts, actsm, X):
        """Compute value of objective function/Lagrangian averaged over batch."""
        errorterm = np.sum(self.compute_errors(acts,actsm, X))
        thetarep = np.repeat(self.theta[:,np.newaxis], self.batch_size,axis=1)
        rateterm = -np.sum(thetarep*(acts - self.p))
        thetamrep = np.repeat(self.thetam[:,np.newaxis], self.batch_size,axis=1)
        rateterm = rateterm - np.sum(thetamrep*(acts - self.p))
        corrWmatrix = np.dot(np.dot(np.transpose(np.concatenate([acts,actsm])), self.W),np.concatenate([acts,actsm]))
        corrterm = -(1/self.batch_size)*np.trace(corrWmatrix) + np.sum(self.W)*self.p**2
        return (errorterm*self.beta/2 + rateterm*self.gamma + corrterm*self.alpha)/self.batch_size


class MirrorSignSAIL(MirrorSAIL):
    """MirrorSAIL but each pair's collective firing rate
    must be p, rather than each unit's rate.
    So each pair has one threshold."""

    def learn(self, X, acts, actsm, corrmatrix):
        """Use learning rules to update network parameters."""

        # update feedforward weights with Oja's rule
        sumsquareacts = np.sum(acts*acts, 1)
        sumsquareactsm = np.sum(actsm**2, 1)
        crossterm = np.sum(acts*actsm, 1)
        dQ = (acts-actsm).dot(X.T) - np.diag(sumsquareacts+sumsquareactsm-2*crossterm).dot(self.Q)
        self.Q = self.Q + self.beta*dQ/self.batch_size

        # update lateral weights with Foldiak's rule
        # (inhibition for decorrelation)
        dW = self.alpha*(corrmatrix - self.p**2)
        self.W = self.W + dW
        self.W = self.W - np.diag(np.diag(self.W))  # zero diagonal entries
        self.W[self.W < 0] = 0  # force weights to be inhibitory

        # update thresholds with Foldiak's rule: keep firing rates near target
        dtheta = self.gamma*((np.sum(acts, 1) + np.sum(actsm, 1))/self.batch_size - self.p)
        self.theta = self.theta + dtheta
        self.thetam = self.theta