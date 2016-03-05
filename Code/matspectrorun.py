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

overcompleteness = 0.5
numinput = 200
numunits = int(overcompleteness*numinput)

pathtoaud = '../../../audition/'

stuff = io.loadmat(pathtoaud+'Nicole Code/PCAmatricesnew.mat')
mypca = pca.pca.PCA(dim=200,whiten=True)
mypca.eVectors = stuff['E'].reshape((25,256,200))[:,::-1,:].reshape((6400,200)).T # flip the PC spectrograms upside down
mypca.sValues = np.sqrt(np.diag(np.abs(stuff['D1'])))
mypca.sValues = mypca.sValues[::-1]
mypca.mean_vec = np.zeros(6400)
mypca.ready=True
origshape = (25,256)
spectros = io.loadmat(pathtoaud+'Nicole Code/dMatPCAnew.mat')['dMatPCA'].T

net = SAILnet.SAILnet(spectros, 'spectro', origshape, ninput=numinput, nunits=numunits, pca=mypca, niter = 50, delay = 0)
alpha0, beta0, gamma0 = net.alpha, net.beta, net.gamma

if overcompleteness == 0.5:
    savedir = pathtoaud+'Results/halfOC/newprep/'
else:
    savedir = pathtoaud+'Results/'+str(overcompleteness)+'OC/newprep/'

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

p=.05
net.p=p
net.initialize()
savestr = savedir + 'SAILp' + str(p)


net.load_params(savestr+'.pickle')
#net.alpha, net.beta, net.gamma = .5*alpha0, beta0, gamma0
#net.run(50000)
##
#plt.figure()
#net.sort_dict(allstims=True)
#plt.savefig(savestr + 'usage.png')
#net.save_params(savestr)
#plt.figure()
#net.show_dict(cmap='jet')
#plt.savefig(savestr+'.png')