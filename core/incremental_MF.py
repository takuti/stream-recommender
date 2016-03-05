from base import Base

import numpy as np

class IncrementalMF(Base):
    """Incremental Matrix Factorization
    """

    def __init__(self, n_user, n_item, static_flg=False, k=100, l2_reg=.01, learn_rate=.03):
        self.n_user = n_user
        self.n_item = n_item

        # if True, parameters will not be updated in evaluation
        self.static_flg = static_flg

        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg
        self.learn_rate = learn_rate

        self._Base__clear()

    def _Base__clear(self):
        self.observed = np.zeros((self.n_user, self.n_item))
        self.P = np.random.normal(0., 0.1, (self.n_user, self.k))
        self.Q = np.random.normal(0., 0.1, (self.n_item, self.k))

    def _Base__update(self, d):
        u_index = d['u_index']
        i_index = d['i_index']

        self.observed[u_index, i_index] = 1

        # static baseline; w/o updating the model
        if self.static_flg: return

        u_vec = self.P[u_index]
        i_vec = self.Q[i_index]

        err = 1. - np.inner(u_vec, i_vec)

        next_u_vec = u_vec + 2. * self.learn_rate * (err * i_vec - self.l2_reg_u * u_vec)
        next_i_vec = i_vec + 2. * self.learn_rate * (err * u_vec - self.l2_reg_i * i_vec)
        self.P[u_index] = next_u_vec
        self.Q[i_index] = next_i_vec

    def _Base__recommend(self, d, at=10):
        u_index = d['u_index']

        pred = np.dot(self.P[u_index], self.Q.T)
        scores = np.abs(1. - pred.reshape(self.n_item))

        return self._Base__scores2recos(u_index, scores, at)
