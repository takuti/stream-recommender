from base import Base

import numpy as np


class Random(Base):

    """Random baseline
    """

    def __init__(self, n_user, n_item):
        self.n_user = n_user
        self.n_item = n_item

        self._Base__clear()

    def _Base__clear(self):
        self.observed = np.zeros((self.n_user, self.n_item))

    def _Base__update(self, d, is_batch_train=False):
        return

    def _Base__recommend(self, d, target_i_indices, at=10):
        u_index = d['u_index']
        scores = np.random.rand(self.n_item)

        return self._Base__scores2recos(u_index, scores, target_i_indices, at)
