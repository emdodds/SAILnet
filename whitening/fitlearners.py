from tf_lca import LCALearner
from SAILmods import VarTimeSAILnet as SAILnet
import numpy as np


"""
Extensions of DictLearner that keep track of how well
they have recovered a known sparse model. The data passed in should
be a StimSet.ToySparseSet object.
"""


def make_fit_learner_class(Learner):
    """Given a particular DictLearner class, returns a version of it that
    keeps track of how well it has recovered a known sparse model."""
    class FitLearner(Learner):
        def initialize_stats(self):
            self.modfits = np.array([])
            Learner.initialize_stats(self)

        def store_statistics(self, *args, **kwargs):
            self.modfits = np.append(self.modfits, self.stims.test_fit(self.Q))
            return Learner.store_statistics(self, *args, **kwargs)

        def get_histories(self):
            histories = Learner.get_histories(self)
            histories['modfits'] = self.modfits
            return histories

        def set_histories(self, histories):
            try:
                self.modfits = histories['modfits']
            except KeyError:
                print('Model fit history not available.')
            Learner.set_histories(self, histories)
    return FitLearner


FittingLCA = make_fit_learner_class(LCALearner)
FittingSAILnet = make_fit_learner_class(SAILnet)
