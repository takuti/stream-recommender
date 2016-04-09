from base import Base

import numpy as np


class IncrementalMF(Base):

    """Incremental Matrix Factorization
    """

    def __init__(self, n_user, n_item, is_positive_only=False, is_static=False, k=40, l2_reg=.01, learn_rate=.003):
        self.n_user = n_user
        self.n_item = n_item

        # whether the problem is based on the explicit rating feedback, or positive-only feedback
        self.is_positive_only = is_positive_only

        # if True, parameters will not be updated in evaluation
        self.is_static = is_static

        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg
        self.learn_rate = learn_rate

        self._Base__clear()

    def _Base__clear(self):
        self.observed = np.zeros((self.n_user, self.n_item))
        self.P = np.random.normal(0., 0.1, (self.n_user, self.k))
        self.Q = np.random.normal(0., 0.1, (self.n_item, self.k))

    def _Base__predict(self, d):
        u_vec = self.P[d['u_index']]
        i_vec = self.Q[d['i_index']]

        return np.inner(u_vec, i_vec)

    def _Base__update(self, d, is_batch_train=False):
        # static baseline; w/o updating the model
        if not is_batch_train and self.is_static:
            return

        u_index = d['u_index']
        i_index = d['i_index']

        u_vec = self.P[u_index]
        i_vec = self.Q[i_index]

        err = d['y'] - np.inner(u_vec, i_vec)

        next_u_vec = u_vec + 2. * self.learn_rate * (err * i_vec - self.l2_reg_u * u_vec)
        next_i_vec = i_vec + 2. * self.learn_rate * (err * u_vec - self.l2_reg_i * i_vec)
        self.P[u_index] = next_u_vec
        self.Q[i_index] = next_i_vec

    def _Base__recommend(self, d, target_i_indices, at=10):
        u_index = d['u_index']

        pred = np.dot(self.P[u_index], self.Q.T)
        if self.is_positive_only:
            scores = np.abs(1. - pred.reshape(self.n_item))
        else:
            scores = pred.reshape(self.n_item)

        return self._Base__scores2recos(u_index, scores, target_i_indices, at)
