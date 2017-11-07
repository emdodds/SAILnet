from tf_lca import LCALearner
from SAILmods import VarTimeSAILnet as SAILnet
import numpy as np


"""
Extensions of DictLearner that keep track of how well
they have recovered a known sparse model. The data passed in should
be a StimSet.ToySparseSet object.
"""


class FittingLCA(LCALearner):

    def initialize_stats(self):
        self.modfits = np.array([])
        LCALearner.initialize_stats(self)

    def store_statistics(self, *args, **kwargs):
        self.modfits = np.append(self.modfits, self.stims.test_fit(self.Q))
        return LCALearner.store_statistics(self, *args, **kwargs)

    def get_histories(self):
        histories = LCALearner.get_histories(self)
        histories['modfits'] = self.modfits
        return histories

    def set_histories(self, histories):
        try:
            self.modfits = histories['modfits']
        except KeyError:
            print('Model fit history not available.')
        LCALearner.set_histories(self, histories)


class FittingSAILnet(SAILnet):
    def initialize_stats(self):
        self.modfits = np.array([])
        SAILnet.initialize_stats(self)

    def store_statistics(self, *args, **kwargs):
        self.modfits = np.append(self.modfits, self.stims.test_fit(self.Q))
        return SAILnet.store_statistics(self, *args, **kwargs)

    def get_histories(self):
        histories = SAILnet.get_histories(self)
        histories['modfits'] = self.modfits
        return histories

    def set_histories(self, histories):
        try:
            self.modfits = histories['modfits']
        except KeyError:
            print('Model fit history not available.')
        SAILnet.set_histories(self, histories)
