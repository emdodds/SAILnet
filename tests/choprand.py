# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:18:14 2015

@author: Eric
"""

from os import chdir
chdir("..")

import scipy.io
import SAILnet

npatches = 10**5

patchgen = SAILnet.SAILnet()
patchgen.batch_size = npatches
patches = patchgen.randpatches()

scipy.io.savemat("manyunrolledpatches.mat", {"patches": patches})

