from base import Base

import numpy as np


class IncrementalBiasedMF(Base):

    """Biased Incremental MF as one specific case of factorization machines; no context
    """

    def __init__(self, n_user, n_item, k=40, l2_reg_w0=.01, l2_reg_w=.01, l2_reg_V=.01, learn_rate=.003):
        self.n_user = n_user
        self.n_item = n_item

        self.k = k
        self.l2_reg_w0 = l2_reg_w0
        self.l2_reg_w = l2_reg_w
        self.l2_reg_V = l2_reg_V
        self.learn_rate = learn_rate

        self.p = n_user + n_item

        self._Base__clear()

    def _Base__clear(self):
        self.observed = np.zeros((self.n_user, self.n_item))
        self.w0 = 0.
        self.w = np.zeros(self.p)
        self.V = np.random.normal(0., 0.1, (self.p, self.k))
        self.prev_w0 = float('inf')
        self.prev_w = np.array([])

    def _Base__update(self, d, is_batch_train=False):
        u_index = d['u_index']
        i_index = d['i_index']

        u = u_index
        i = self.n_user + i_index

        # make prediction and compute current error
        pred = np.inner(self.V[u], self.V[i]) + self.w0 + self.w[u] + self.w[i]
        err = 1. - pred

        # update regularization parameters
        if self.prev_w0 != float('inf') and self.prev_w.size != 0:
            self.l2_reg_w0 = max(0., self.l2_reg_w0 + 4. * self.learn_rate * (err * self.learn_rate * self.prev_w0))
            self.l2_reg_w = max(0., self.l2_reg_w + 4. * self.learn_rate * (err * self.learn_rate * (self.prev_w[u] + self.prev_w[i])))

        # update w0 with keeping the previous w0 to optimize it
        self.prev_w0 = self.w0
        self.w0 = self.w0 + 2. * self.learn_rate * (err * 1. - self.l2_reg_w0 * self.w0)

        # update w
        self.prev_w = np.empty_like(self.w)
        self.prev_w[:] = self.w
        self.w[u] = self.w[u] + 2. * self.learn_rate * (err * 1. - self.l2_reg_w * self.w[u])
        self.w[i] = self.w[i] + 2. * self.learn_rate * (err * 1. - self.l2_reg_w * self.w[i])

        # update V
        next_u_vec = self.V[u] + 2. * self.learn_rate * (err * self.V[i] - self.l2_reg_V * self.V[u])
        next_i_vec = self.V[i] + 2. * self.learn_rate * (err * self.V[u] - self.l2_reg_V * self.V[i])
        self.V[u] = next_u_vec
        self.V[i] = next_i_vec

    def _Base__recommend(self, d, target_i_indices, at=10):
        u_index = d['u_index']
        i_offset = self.n_user

        pred = np.dot(self.V[u_index], self.V[i_offset:].T) + self.w0 + self.w[u_index] + self.w[i_offset:]
        scores = np.abs(1. - pred.reshape(self.n_item))

        return self._Base__scores2recos(u_index, scores, target_i_indices, at)
