# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:20:46 2015

@author: Eric
"""

import SAILnet


ntimes = 25
nfreqs = 256
overcompleteness = 1
numunits = int(overcompleteness*200)
net = SAILnet.SAILnet(imagefilename = "processedspeech2015_6_30.mat",
                      imagevarname = "processedspeech", datatype = "spectro",
                      npixels = ntimes*nfreqs, nunits = numunits)
#net.run()
net.load_params("params_speech2015_6_30.pickle")
net.showrfs()