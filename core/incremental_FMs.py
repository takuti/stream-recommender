from base import Base

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot

class IncrementalFMs(Base):
    """Incremental Factorization Machines
    """

    def __init__(self, n_user, n_item, contexts, k=100, l2_reg_w0=.01, l2_reg_w=.01, l2_reg_V=.01, learn_rate=.03):
        self.n_user = n_user
        self.n_item = n_item
        self.contexts = contexts
        self.n_context = len(contexts)

        self.p = n_user + n_item
        if 'dt' in contexts: self.p += 1
        if 'genre' in contexts: self.p += 18 # 18 genres
        if 'demographics' in contexts: self.p += 23 # 1 for M/F, 1 for age, 21 for occupation(0-20)

        # parameter settings
        self.k = k
        self.l2_reg_w0 = l2_reg_w0
        self.l2_reg_w = l2_reg_w
        self.l2_reg_V = l2_reg_V
        self.learn_rate = learn_rate

        self._Base__clear()

    def _Base__clear(self):
        self.observed = np.zeros((self.n_user, self.n_item))
        self.w0 = 0.
        self.w = np.zeros(self.p)
        self.V = np.random.normal(0., 0.1, (self.p, self.k))
        self.prev_w0 = float('inf')
        self.prev_w = np.array([])

    def _Base__update(self, d):
        """Update the model parameters based on the given vector-value pair.
        """

        u_index = d['u_index']
        i_index = d['i_index']

        self.observed[u_index, i_index] = 1

        x = np.zeros(self.n_user + self.n_item)
        x[u_index] = x[self.n_user + i_index] = 1

        for c in self.contexts:
            x = np.append(x, d[c])

        x_vec = np.array([x]).T # p x 1
        interaction = np.sum(np.dot(self.V.T, x_vec) ** 2 - np.dot(self.V.T ** 2, x_vec ** 2)) / 2.
        pred = self.w0 + np.inner(self.w, x) + interaction

        err = 1. - pred

        # Updating regularization parameters
        if self.prev_w0 != float('inf') and self.prev_w.size != 0:
            self.l2_reg_w0 = max(0., self.l2_reg_w0 + 4. * self.learn_rate * (err * self.learn_rate * self.prev_w0))
            self.l2_reg_w = max(0., self.l2_reg_w + 4. * self.learn_rate * (err * self.learn_rate * np.inner(x, self.prev_w)))

        # Updating model parameters
        self.prev_w0 = self.w0
        self.w0 = self.w0 + 2. * self.learn_rate * (err * 1. - self.l2_reg_w0 * self.w0)

        # x_u and x_i are 1.0
        self.prev_w = np.empty_like(self.w)
        self.prev_w[:] = self.w

        prev_V = np.empty_like(self.V)
        prev_V[:] = self.V

        for pi in xrange(self.p):
            if x[pi] == 0.: continue

            self.w[pi] = self.prev_w[pi] + 2. * self.learn_rate * (err * x[pi] - self.l2_reg_w * self.prev_w[pi])

            g = err * x[pi] * (np.dot(np.array([x]), prev_V) - x[pi] * prev_V[pi])
            self.V[pi] = prev_V[pi] + 2. * self.learn_rate * (g - self.l2_reg_V * prev_V[pi])

    def _Base__recommend(self, d, at=10):
        # i_mat is (p, n_item) for all possible pairs of the user (d['u_index']) and all items
        i_mat = self.__create_i_mat(d)

        # Matrix A and B should be dense (numpy array; rather than scipy CSR matrix) because V is dense.
        A = safe_sparse_dot(self.V.T, i_mat)
        A = A ** 2

        sq_i_mat = i_mat.copy()
        sq_i_mat.data[:] = sq_i_mat.data ** 2
        B = safe_sparse_dot(self.V.T ** 2, sq_i_mat)

        interaction = np.sum(A - B, 0)
        interaction /= 2. # (n_item,)

        pred = self.w0 + safe_sparse_dot(self.w, i_mat, dense_output=True) + interaction

        scores = np.abs(1. - (pred[:self.n_item] + np.sum(pred[self.n_item:])))
        return self._Base__scores2recos(d['u_index'], scores, at)

    def __create_i_mat(self, d):
        # (n_user, n_item); user of d's row has 1s
        row_upper = np.ones(self.n_item) * d['u_index']
        col_upper = np.arange(self.n_item)
        data_upper = np.ones(self.n_item)

        # (n_item, n_item); identity matrix
        row_lower = np.arange(self.n_user, self.n_user + self.n_item)
        col_lower = np.arange(self.n_item)
        data_lower = np.ones(self.n_item)

        # concat
        row = np.append(row_upper, row_lower)
        col = np.append(col_upper, col_lower)
        data = np.append(data_upper, data_lower)

        # for each context, extend the cancatenated arrays
        for ctx_index in xrange(self.n_context):
            ctx = self.contexts[ctx_index]

            row_ctx = np.ones(self.n_item) * (self.n_user + self.n_item + ctx_index)
            col_ctx = np.arange(self.n_item)
            data_ctx = np.ones(self.n_item) * d[ctx]

            row = np.append(row, row_ctx)
            col = np.append(col, col_ctx)
            data = np.append(data, data_ctx)

        return sp.csr_matrix((data, (row, col)), shape=(self.p, self.n_item))
