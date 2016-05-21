from base import Base

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot


class IncrementalFMs(Base):

    """Incremental Factorization Machines
    """

    def __init__(
            self, contexts, k=6, l2_reg_w0=.01, l2_reg_w=.01, l2_reg_V=30., learn_rate=.003):

        self.contexts = contexts
        self.p = contexts['user'] + contexts['item']

        self.k = k
        self.l2_reg_w0 = l2_reg_w0
        self.l2_reg_w = l2_reg_w
        self.l2_reg_V = np.ones(k) * l2_reg_V
        self.learn_rate = learn_rate

        self._Base__clear()

    def _Base__clear(self):
        self.n_user = 0
        self.users = {}

        self.n_item = 0
        self.items = {}

        self.i_mat = sp.csr_matrix([])

        # initial parameters from Gaussian
        self.w0 = np.random.normal(0., 0.1)
        self.w = np.random.normal(0., 0.1, self.p)
        self.V = np.random.normal(0., 0.1, (self.p, self.k))

        # to keep the last parameters for adaptive regularization
        self.prev_w0 = self.w0
        self.prev_w = self.w.copy()
        self.prev_V = self.V.copy()

    def _Base__check(self, d):
        u_index = d['u_index']
        if u_index not in self.users:
            self.users[u_index] = {'observed': set()}
            self.n_user += 1

        i_index = d['i_index']
        if i_index not in self.items:
            self.items[i_index] = {}
            self.n_item += 1
            i = sp.csr_matrix(np.array([d['item']]).T)
            self.i_mat = i if self.i_mat.size == 0 else sp.csr_matrix(sp.hstack((self.i_mat, i)))

    def _Base__update(self, d, is_batch_train=False):
        x = np.concatenate((d['user'], d['item']))

        x_vec = np.array([x]).T  # p x 1
        interaction = np.sum(np.dot(self.V.T, x_vec) ** 2 - np.dot(self.V.T ** 2, x_vec ** 2)) / 2.
        pred = self.w0 + np.inner(self.w, x) + interaction

        # compute current error
        err = d['y'] - pred

        # update regularization parameters
        coeff = 4. * self.learn_rate * err * self.learn_rate

        self.l2_reg_w0 = max(0., self.l2_reg_w0 + coeff * self.prev_w0)
        self.l2_reg_w = max(0., self.l2_reg_w + coeff * np.inner(x, self.prev_w))

        dot_v = np.dot(x_vec.T, self.V).reshape((self.k,))  # (k, )
        dot_prev_v = np.dot(x_vec.T, self.prev_V).reshape((self.k,))  # (k, )
        s_duplicated = np.dot((x_vec.T ** 2), self.V * self.prev_V).reshape((self.k,))  # (k, )
        self.l2_rev_V = np.maximum(np.zeros(self.k), self.l2_reg_V + coeff * (dot_v * dot_prev_v - s_duplicated))

        # update w0 with keeping the previous value
        self.prev_w0 = self.w0
        self.w0 = self.w0 + 2. * self.learn_rate * (err * 1. - self.l2_reg_w0 * self.w0)

        # keep the previous w for auto-parameter optimization
        self.prev_w = np.empty_like(self.w)
        self.prev_w[:] = self.w

        # keep the previous V
        self.prev_V = np.empty_like(self.V)
        self.prev_V[:] = self.V

        # update w and V
        prod = np.dot(np.array([x]), self.prev_V)  # (1, p) and (p, k) => (1, k)
        for pi in xrange(self.p):
            if x[pi] == 0.:
                continue

            self.w[pi] = self.prev_w[pi] + 2. * self.learn_rate * (err * x[pi] - self.l2_reg_w * self.prev_w[pi])

            g = err * x[pi] * (prod - x[pi] * self.prev_V[pi])
            self.V[pi] = self.prev_V[pi] + 2. * self.learn_rate * (g - self.l2_reg_V * self.prev_V[pi])

    def _Base__recommend(self, d, target_i_indices, at=10):
        # i_mat is (n_item_context, n_item) for all possible items
        # extract only target items
        i_mat = self.i_mat[:, target_i_indices]

        n_target = len(target_i_indices)

        # u_mat will be (n_user + n_user_context, n_item) for the target user
        u_vec = np.array([d['user']]).T
        u_mat = sp.csr_matrix(np.repeat(u_vec, n_target, axis=1))

        # stack them into (p, n_item) matrix
        mat = sp.vstack((u_mat, i_mat))

        # Matrix A and B should be dense (numpy array; rather than scipy CSR matrix) because V is dense.
        V = sp.csr_matrix(self.V)
        A = safe_sparse_dot(V.T, mat)
        A.data[:] = A.data ** 2

        sq_mat = mat.copy()
        sq_mat.data[:] = sq_mat.data ** 2
        sq_V = V.copy()
        sq_V.data[:] = sq_V.data ** 2
        B = safe_sparse_dot(sq_V.T, sq_mat)

        interaction = (A - B).sum(axis=0)
        interaction /= 2.  # (1, n_item); numpy matrix form

        pred = self.w0 + safe_sparse_dot(self.w, mat, dense_output=True) + interaction

        scores = np.abs(1. - np.ravel(pred))

        return self._Base__scores2recos(scores, target_i_indices, at)
