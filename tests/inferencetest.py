# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:50:29 2015

@author: Eric
"""

import SAILnet

net = SAILnet.SAILnet()
net.load_params("imPC7_21.pickle")
someX = net.randpatches()
net.compute_activities(someX, True)