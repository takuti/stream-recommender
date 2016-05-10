from base import Base

import numpy as np


class Random(Base):

    """Random baseline
    """

    def __init__(self, n_item):
        self.n_item = n_item

        self._Base__clear()

    def _Base__clear(self):
        self.n_user = 0
        self.users = {}

    def _Base__check(self, d):
        u_index = d['u_index']

        if u_index not in self.users:
            self.users[u_index] = {'observed': set()}
            self.n_user += 1

    def _Base__update(self, d, is_batch_train=False):
        return

    def _Base__recommend(self, d, target_i_indices, at=10):
        scores = np.random.rand(len(target_i_indices))
        return self._Base__scores2recos(scores, target_i_indices, at)
