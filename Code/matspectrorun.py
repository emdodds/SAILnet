# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 00:30:43 2015

@author: Eric
"""
import argparse
import scipy.io as io
import SAILnet
import numpy as np
import pca.pca

import matplotlib.pyplot as plt
plt.ioff()

parser = argparse.ArgumentParser(description="Run SAILnet with given parameters.")
parser.add_argument('-o', '--overcompleteness', default=4, type=float)
parser.add_argument('-f', '--datafolder', default='../audition/Nicole Code/', type=str)
parser.add_argument('-r', '--resultsfolder', default='../audition/Results/',type=str)
parser.add_argument('-s', '--datasuffix', default='new', type=str)
parser.add_argument('-p', default = .005, type=float)
args=parser.parse_args()

datafolder = args.datafolder
resultsfolder = args.resultsfolder
oc = args.overcompleteness
datasuffix = args.datasuffix

numinput = 200
numunits = int(oc*numinput)

stuff = io.loadmat(datafolder+'PCAmatrices'+datasuffix+'.mat')
mypca = pca.pca.PCA(dim=200,whiten=True)
mypca.eVectors = stuff['E'].reshape((25,256,200))[:,::-1,:].reshape((6400,200)).T # flip the PC spectrograms upside down
mypca.sValues = np.sqrt(np.diag(np.abs(stuff['D1'])))
mypca.sValues = mypca.sValues[::-1]
mypca.mean_vec = np.zeros(6400)
mypca.ready=True
origshape = (25,256)
spectros = io.loadmat(datafolder+'dMatPCA'+datasuffix+'.mat')['dMatPCA'].T

net = SAILnet.SAILnet(spectros, 'spectro', origshape, ninput=numinput, nunits=numunits, pca=mypca)
alpha0, beta0, gamma0 = net.alpha, net.beta, net.gamma

#if overcompleteness == 0.5:
#    savedir = pathtoaud+'Results/halfOC/newprep/'
#else:
#    savedir = pathtoaud+'Results/'+str(overcompleteness)+'OC/newprep/'

p=args.p
net.p=p
net.initialize()
savestr = str(oc) + 'OCSAILp'+str(p) + datasuffix


net.save_params(savestr+'.pickle')
net.run(10000)
net.run(200000, rate_decay=.99995)
net.sort_dict(allstims=True, plot=False)
net.save_params()

#net.errorhistory = net.errorhistory/numunits/numinput
#net.plotter.save_plots(savestr, fastsort=True, savesorted=True)