# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:02:44 2016

@author: Eric
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    print('Failed to import matplotlib.')


class Plotter(object):

    def __init__(self, net):
        self.net = net

    def show_network(self):
        """
        Plot current values of weights, thresholds, and time-averaged firing
        correlations.
        """
        net = self.net
        plt.subplot(2, 2, 1)
        plt.imshow(net.W, cmap="gray", interpolation="nearest", aspect='auto')
        plt.colorbar()
        plt.title("Inhibitory weights")

        plt.subplot(2, 2, 2)
        C = net.corrmatrix_ave - np.outer(net.L1acts, net.L1acts)
        C = C - np.diag(np.diag(C))
        plt.imshow(C, cmap="gray", interpolation="nearest", aspect='auto')
        plt.colorbar()
        plt.title("Moving time-averaged covariance matrix")

        plt.subplot(2, 2, 3)
        # The first point is usually huge compared to everything else, so just ignore it
        plt.plot(net.objhistory[1:], 'b', net.errorhist[1:], 'g')
        plt.title("History of objective function (blue) and mean squared error")

        plt.subplot(2, 2, 4)
        plt.bar(np.arange(net.theta.size), net.theta)
        plt.title(r"Thresholds $\theta$")

    def visualize(self, cmap='gray'):
        """Display visualizations of network parameters."""
        plt.figure()
        self.show_network()
        plt.figure()
        self.net.show_dict(cmap=cmap)
        plt.show()

    def inference_plots(self, errors, activities, savestr = None):
        plt.figure()
        plt.plot(errors)
        plt.title('Mean squared error vs inference time step')
        if savestr is not None:
            plt.savefig(savestr+'inferror.png', bbox_inches='tight')
        plt.figure()
        plt.plot(activities)
        plt.title('Activity per stimulus and per unit, vs inference time step')
        if savestr is not None:
            plt.savefig(savestr+'avgact.png', bbox_inches='tight')

    def save_plots(self, savestr, X=None, subset=100, fastsort=False, savesorted=True):
        """
        Save five figures showing the behavior of the network:
        1)  L0 or L1 (for fastsort = True) usage after sorting by the same
        2) The dictionary elements tiled in a grid. A random subset of size subset is selected unless subset=None
        3-4) See inference_plots
        5) See show_network
        """
        net=self.net
        plt.figure()
        if fastsort:
            net.fast_sort()
        else:
            net.sort_dict(allstims=True)
        plt.savefig(savestr + 'usage.png', bbox_inches='tight')
        if savesorted:
            net.save_params()
        plt.figure()
        net.show_dict(subset=subset)
        plt.savefig(savestr+'.png', bbox_inches='tight')
        X = X or net.stims.rand_stim()
        net.infer(X, infplot=True, savestr=savestr)
        self.show_network()
        plt.savefig(savestr+'netparams.png', bbox_inches='tight')
