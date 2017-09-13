import numpy as np
import scipy.io as io
import pickle
import argparse
import os
import sys
import StimSet
sys.path.append('../Code/')
import SAILmods
import SAILnet

parser = argparse.ArgumentParser()
parser.add_argument('--scaled', dest='scaled', action='store_true')
parser.add_argument('--not-scaled', dest='scaled', action='store_false')
parser.add_argument('-g', '--gain', default=2, type=float)
parser.add_argument('--load', dest='load', action='store_true')
parser.add_argument('-w', '--whiten', dest='whiten', action='store_true')
parser.add_argument('--nonneg', dest='nonneg', action='store_true')
parser.add_argument('-p', '--firing_rate', default=0.05, type=float)
parser.add_argument('--oc', default=1, type=float)
parser.add_argument('--keep_only_fit', dest='keep_only_fit', action='store_true')
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--beta', default=0.01, type=float)
parser.add_argument('--gamma', default=0.1, type=float)
parser.set_defaults(keep_only_fit=False)
parser.set_defaults(scaled=True)
parser.set_defaults(load=False)
parser.set_defaults(whiten=False)
parser.set_defaults(nonneg=False)
args = parser.parse_args()

kwargs = {'p': args.firing_rate,
          'theta0': 1.5}

if args.scaled:
    Net = SAILmods.VarTimeSAILnet
    prefix = 'scaled' + str(args.gain)
    kwargs['gain_rate'] = 0.0
    kwargs['gain'] = args.gain
else:
    Net = SAILnet.SAILnet
    prefix = ''
paramfile = 'toy'+prefix+'SAIL'+str(args.oc)+'oc'+str(args.firing_rate)+'p.pickle'

kwargs['alpha'] = args.alpha
kwargs['beta'] = args.beta
kwargs['gamma'] = args.gamma

toy = StimSet.ToySparseSet(nonneg=args.nonneg, scale=0.05)

numinput = 200
numunits = int(numinput*args.oc)
net = Net(data=toy, nunits=numunits,
          paramfile=paramfile,
          **kwargs)


def track_fit(net, nsteps=100000, oldfits=None):
    fit = np.zeros(nsteps)
    self = net
    for tt in range(nsteps):
        X = self.stims.rand_stim()

        acts = self.infer(X)
        errors = np.mean(self.compute_errors(acts, X))
        corrmatrix = self.store_statistics(acts, errors)
        self.objhistory = np.append(self.objhistory,
                                    self.compute_objective(acts, X))

        self.learn(X, acts, corrmatrix)

        fit[tt] = toy.test_fit(net.Q)

        if tt % 50 == 0:
            print("Trial number: " + str(tt))
            if tt % 5000 == 0:
                # save progress
                print("Saving progress...")
                self.save()
                print("Done. Continuing to run...")
                np.save(net.paramfile+'fit'+'.npy', fit)
    self.save()
    np.save(net.paramfile+'fit'+'.npy', fit)

    if oldfits is not None:
        return np.concatenate([oldfits, fit])
    else:
        return fit


track_fit(net, 100000)
