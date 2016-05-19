from base import Base

import numpy as np


class Popular(Base):

    """Popularity (non-personalized) baseline
    """

    def __init__(self, n_item):
        self.n_item = n_item
        self.freq = np.zeros(n_item)

        self._Base__clear()

    def _Base__clear(self):
        self.n_user = 0
        self.users = {}

    def _Base__check(self, d):
        self.freq[d['i_index']] += 1

        u_index = d['u_index']

        if u_index not in self.users:
            self.users[u_index] = {'observed': set()}
            self.n_user += 1

    def _Base__update(self, d, is_batch_train=False):
        return

    def _Base__recommend(self, d, target_i_indices, at=10):
        sorted_indices = np.argsort(self.freq[target_i_indices])[::-1]
        return target_i_indices[sorted_indices][:at]
