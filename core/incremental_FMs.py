from base import Base

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot


class IncrementalFMs(Base):

    """Incremental Factorization Machines
    """

    def __init__(
            self, n_item, samples, contexts, k=40, l2_reg_w0=.01, l2_reg_w=.01, l2_reg_V=30., learn_rate=.003):

        self.n_item = n_item

        self.contexts = contexts
        self.p = np.sum(contexts.values())

        # create item matrices which has contexts of each item in rows
        self.i_mat = self.__create_i_mat(samples)

        self.k = k
        self.l2_reg_w0 = l2_reg_w0
        self.l2_reg_w = l2_reg_w
        self.l2_reg_V = np.ones(k) * l2_reg_V
        self.learn_rate = learn_rate

        self._Base__clear()

    def _Base__clear(self):
        self.n_user = 0
        self.users = {}

        # initial parameters from Gaussian
        self.w0 = np.random.normal(0., 0.1, 1)
        self.w = np.random.normal(0., 0.1, self.p)
        self.V = np.random.normal(0., 0.1, (self.p, self.k))

        # to keep the last parameters for adaptive regularization
        self.prev_w0 = float('inf')
        self.prev_w = np.array([])
        self.prev_V = np.array([])

    def _Base__check(self, d):
        u_index = d['u_index']

        if u_index not in self.users:
            self.users[u_index] = {'observed': set()}
            self.n_user += 1

    def _Base__update(self, d, is_batch_train=False):
        # create a sample vector and make prediction
        x = d['user']
        x = np.append(x, d['item'])
        x = np.append(x, d['dt'])

        x_vec = np.array([x]).T  # p x 1
        interaction = np.sum(np.dot(self.V.T, x_vec) ** 2 - np.dot(self.V.T ** 2, x_vec ** 2)) / 2.
        pred = self.w0 + np.inner(self.w, x) + interaction

        # compute current error
        err = d['y'] - pred

        # update regularization parameters
        if self.prev_w0 != float('inf') and self.prev_w.size != 0 and self.prev_V.size != 0:
            self.l2_reg_w0 = max(0., self.l2_reg_w0 + 4. * self.learn_rate * (err * self.learn_rate * self.prev_w0))
            self.l2_reg_w = max(0., self.l2_reg_w + 4. * self.learn_rate * (err * self.learn_rate * np.inner(x, self.prev_w)))

            dot_v = np.dot(x_vec.T, self.V).reshape((self.k,))  # (k, )
            dot_prev_v = np.dot(x_vec.T, self.prev_V).reshape((self.k,))  # (k, )
            s_duplicated = np.dot((x_vec.T ** 2), self.V * self.prev_V).reshape((self.k,))  # (k, )
            self.l2_rev_V = np.maximum(np.zeros(self.k), self.l2_reg_V + 4. * self.learn_rate * (err * self.learn_rate * (dot_v * dot_prev_v - s_duplicated)))

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

        # u_mat will be (n_user_context, n_item) for the target user
        u_mat = np.repeat(np.array([d['user']]).T, self.n_item, axis=1)

        # dt_mat will be (1, n_item) for the target user
        dt_mat = np.repeat(d['dt'], self.n_item)

        # stack them into (p, n_item) matrix
        mat = sp.csr_matrix(np.vstack((u_mat, self.i_mat, dt_mat))[:, target_i_indices])

        # Matrix A and B should be dense (numpy array; rather than scipy CSR matrix) because V is dense.
        A = safe_sparse_dot(self.V.T, mat)
        A = A ** 2

        sq_mat = mat.copy()
        sq_mat.data[:] = sq_mat.data ** 2
        B = safe_sparse_dot(self.V.T ** 2, sq_mat)

        interaction = np.sum(A - B, 0)
        interaction /= 2.  # (n_item,)

        pred = self.w0 + safe_sparse_dot(self.w, mat, dense_output=True) + interaction
        scores = np.abs(1. - (pred[:self.n_item] + np.sum(pred[self.n_item:])))

        return self._Base__scores2recos(scores, target_i_indices, at)

    def __create_i_mat(self, samples):
        """Create an item matrix which has contexts of each item in rows.

        Args:
            samples (list of dict): Each sample has an item vector.

        Returns:
            numpy array (n_item_context, n_item): Column is an item vector.

        """
        i_mat = np.zeros((self.n_item, self.contexts['item']))
        observed = np.zeros(self.n_item)

        for d in samples:
            i_index = d['i_index']

            if observed[i_index] == 1:
                continue

            i_mat[i_index, :] = d['item']

        return i_mat.T
