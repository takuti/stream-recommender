from .base import Base

import numpy as np


class IncrementalMF(Base):

    """Incremental Matrix Factorization
    """

    def __init__(self, is_static=False, k=40, l2_reg=.01, learn_rate=.003):
        # if True, parameters will not be updated in evaluation
        self.is_static = is_static

        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg
        self.learn_rate = learn_rate

        self._Base__clear()

    def _Base__clear(self):
        self.n_user = 0
        self.users = {}

        self.n_item = 0
        self.items = {}

        self.Q = np.array([])

    def _Base__check(self, d):
        u_index = d['u_index']
        if u_index not in self.users:
            self.users[u_index] = {'vec': np.random.normal(0., 0.1, self.k), 'observed': set()}
            self.n_user += 1

        i_index = d['i_index']
        if i_index not in self.items:
            self.items[i_index] = {}
            self.n_item += 1
            i = np.random.normal(0., 0.1, (1, self.k))
            self.Q = i if self.Q.size == 0 else np.concatenate((self.Q, i))

    def _Base__update(self, d, is_batch_train=False):
        # static baseline; w/o updating the model
        if not is_batch_train and self.is_static:
            return

        u_index = d['u_index']
        i_index = d['i_index']

        u_vec = self.users[u_index]['vec']
        i_vec = self.Q[i_index]

        err = d['y'] - np.inner(u_vec, i_vec)

        next_u_vec = u_vec + 2. * self.learn_rate * (err * i_vec - self.l2_reg_u * u_vec)
        next_i_vec = i_vec + 2. * self.learn_rate * (err * u_vec - self.l2_reg_i * i_vec)
        self.users[u_index]['vec'] = next_u_vec
        self.Q[i_index] = next_i_vec

    def _Base__recommend(self, d, target_i_indices):
        pred = np.dot(self.users[d['u_index']]['vec'], self.Q[target_i_indices, :].T)
        scores = np.abs(1. - pred.flatten())

        return self._Base__scores2recos(scores, target_i_indices)
