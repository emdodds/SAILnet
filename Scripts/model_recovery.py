import numpy as np
import pickle
import argparse
import os
from pathlib import Path
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
parser.add_argument('--noise', default=0.1, type=float)
parser.add_argument('--desphere', default=0.0, type=float) # 0 is leave whitened, 1 gives spectrum of natural images
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
    prefix = 'sc' + str(args.gain)+'ds'+str(args.desphere)
    kwargs['gain_rate'] = 0.0
    kwargs['gain'] = args.gain
else:
    Net = SAILnet.SAILnet
    prefix = ''
paramfile = 'toy'+prefix+'SAIL'+str(args.oc)+'oc'+str(args.firing_rate)+'p.pickle'
num = 0
while os.path.exists(paramfile) and not args.load:
    paramfile = 'toy'+prefix+'SAIL'+str(args.oc)+'oc'+str(args.firing_rate)+'p'+str(num)+'.pickle'
    num += 1
# create file to reserve its name
Path(paramfile).touch()

kwargs['alpha'] = args.alpha
kwargs['beta'] = args.beta
kwargs['gamma'] = args.gamma

numinput = 256
datascale = 0.05
toy = StimSet.ToySparseSet(dim=numinput, nonneg=args.nonneg, scale=datascale,
                           noise=datascale*args.noise, white=False)

if args.desphere > 0:
    with open('/global/home/users/edodds/vision/Data/vh32_256PCA.pickle', 'rb') as f:
        impca, imshape = pickle.load(f)
    toy.data = toy.data.dot(np.diag(np.power(impca.sValues[:numinput], args.desphere)))

toy.data /= toy.data.std()

numunits = int(numinput*args.oc)
net = Net(data=toy, nunits=numunits,
          paramfile=paramfile,
          **kwargs)


def track_fit(net, nsteps=100000, oldfits=None):
    fit = np.zeros(nsteps)
    for tt in range(nsteps):
        X = net.stims.rand_stim()

        acts = net.infer(X)
        errors = np.mean(net.compute_errors(acts, X))
        corrmatrix = net.store_statistics(acts, errors)
        net.objhistory = np.append(net.objhistory,
                                   net.compute_objective(acts, X))

        net.learn(X, acts, corrmatrix)

        fit[tt] = toy.test_fit(net.Q)

        if tt % 50 == 0:
            print("Trial number: " + str(tt))
            if tt % 5000 == 0:
                # save progress
                print("Saving progress...")
                net.save()
                print("Done. Continuing to run...")
                np.save(net.paramfile+'fit'+'.npy', fit)
    net.save()
    np.save(net.paramfile+'fit'+'.npy', fit)

    if oldfits is not None:
        return np.concatenate([oldfits, fit])
    else:
        return fit


track_fit(net, 100000)
