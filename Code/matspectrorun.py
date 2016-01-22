# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 00:30:43 2015

@author: Eric
"""

import scipy.io as io
import SAILnet
import numpy as np
import pca.pca

import matplotlib.pyplot as plt
plt.ioff()

overcompleteness = 8
numinput = 200
numunits = int(overcompleteness*numinput)

pathtoaud = '../../../audition/'

stuff = io.loadmat(pathtoaud+'Nicole Code/PCAmatrices3.mat')
mypca = pca.pca.PCA(dim=200,whiten=True)
mypca.eVectors = stuff['E'].reshape((25,256,200))[:,::-1,:].reshape((6400,200)).T # flip the PC spectrograms upside down
mypca.sValues = np.sqrt(np.diag(np.abs(stuff['D1'])))
mypca.sValues = mypca.sValues[::-1]
mypca.mean_vec = np.zeros(6400)
mypca.ready=True
origshape = (25,256)
spectros = io.loadmat(pathtoaud+'Nicole Code/dMatPCA3.mat')['dMatPCA'].T

net = SAILnet.SAILnet(spectros, 'spectro', origshape, ninput=numinput, nunits=numunits, pca=mypca)
alpha0, beta0, gamma0 = net.alpha, net.beta, net.gamma


savedir = pathtoaud+'Results/8OC/newprep/'

#pvalues = [0.05,0.1,0.2]
#for p in pvalues:
#    net.p = p
#    net.alpha, net.beta, net.gamma = alpha0, beta0, gamma0
#    net.initialize()
#    
#    savestr = savedir + 'SAILp' + str(p)
#    net.save_params(savestr + '.pickle')
#    
#    net.run(ntrials = 50000, rate_decay = .9999)
#    plt.figure()
#    net.sort_dict(allstims=True)
#    plt.savefig(savestr + 'usage.png')
#    net.save_params()
#    plt.figure()
#    net.show_dict(cmap='jet')
#    plt.savefig(savestr+'.png')

p=0.05
net.p=p
net.initialize()
savestr = savedir + 'SAILp' + str(p)
#net.save_params(savestr + '.pickle')
net.load_params(savestr+'.pickle') # TODO: uncomment when running old things
#net.alpha, net.beta, net.gamma = alpha0, beta0, gamma0
#net.run(100000,rate_decay=.99995)
#net.run(100000)
#
#plt.figure()
#net.sort_dict(allstims=True)
#plt.savefig(savestr + 'usage.png')
#net.save_params(savestr)
#plt.figure()
#net.show_dict(cmap='jet')
#plt.savefig(savestr+'.png')