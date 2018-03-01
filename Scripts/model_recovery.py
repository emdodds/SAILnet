import numpy as np
import pickle
import argparse
import os
from pathlib import Path
import sys
import StimSet
import SAILmods
sys.path.append('../whitening/')
import fitlearners

parser = argparse.ArgumentParser()
parser.add_argument('--learner', default='SAILnet', type=str)
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
parser.add_argument('--dim', default=256, type=int)
parser.set_defaults(keep_only_fit=False)
parser.set_defaults(load=False)
parser.set_defaults(whiten=False)
parser.set_defaults(nonneg=False)
args = parser.parse_args()


if args.learner == 'SAILnet':
    Net = fitlearners.FittingSAILnet
    prefix = 'sail' + str(args.gain)
    kwargs = {'p': args.firing_rate,
              'theta0': 1.5,
              'gain_rate': 0.0,
              'gain': args.gain}
elif args.learner == 'LCA':
    Net = fitlearners.FittingLCA
    kwargs = {'lam': args.firing_rate,
              'learnrate': 50.0,
              'infrate': 0.1,
              'niter': 200,
              'seek_snr_rate': 0.0,
              'threshfunc': 'soft'}
    prefix = 'lca'
elif args.learner == 'LCALocal':
    Net = fitlearners.make_fit_learner_class(SAILmods.LCALocalLearner)
    prefix = 'lcalocal'
    kwargs = {'p': args.firing_rate,
              'theta0': 1.5,
              'niter': 200,
              'alpha': 0.1}
else:
    raise ValueError('Learner class not supported.')

prefix = prefix + 'ds'+str(args.desphere)
paramfile = 'toy'+prefix+str(args.oc)+'oc'+str(args.firing_rate)+'p.pickle'
num = 0
while os.path.exists(paramfile) and not args.load:
    paramfile = 'toy'+prefix+'SAIL'+str(args.oc)+'oc'+str(args.firing_rate)+'p'+str(num)+'.pickle'
    num += 1
# create file to reserve its name
Path(paramfile).touch()

kwargs['alpha'] = args.alpha
kwargs['beta'] = args.beta
kwargs['gamma'] = args.gamma

numinput = args.dim
numsources = int(numinput*args.oc)
toy = StimSet.ToySparseSet(dim=numinput, nsource=numsources, nonneg=args.nonneg,
                           scale=1, noise=args.noise, white=False)

if args.desphere > 0:
    with open('/global/home/users/edodds/vision/Data/vh32_256PCA.pickle', 'rb') as f:
        impca, imshape = pickle.load(f)
    toy.data = toy.data.dot(np.diag(np.power(impca.sValues[:numinput], args.desphere)))

toy.data /= toy.data.std()

numunits = int(numinput*args.oc)
net = Net(data=toy, nunits=numunits,
          paramfile=paramfile,
          store_every=100,
          **kwargs)

net.run(100000)
numruns = 1
while net.modfits[-1] < 0.9 and numruns < 5:
    net.run(100000)
    numruns += 1
net.save()
