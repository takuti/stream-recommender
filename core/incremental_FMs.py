from base import Base

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot


class IncrementalFMs(Base):

    """Incremental Factorization Machines
    """

    def __init__(self, n_user, n_item, is_positive_only, contexts, k=40, l2_reg_w0=.01, l2_reg_w=.01, l2_reg_V=30., learn_rate=.003):
        self.n_user = n_user
        self.n_item = n_item

        self.is_positive_only = is_positive_only

        self.contexts = contexts

        self.p = n_user + n_item
        for ctx, dim in contexts:
            self.p += dim

        self.k = k
        self.l2_reg_w0 = l2_reg_w0
        self.l2_reg_w = l2_reg_w
        self.l2_reg_V = np.ones(k) * l2_reg_V
        self.learn_rate = learn_rate

        self._Base__clear()

    def _Base__clear(self):
        self.observed = np.zeros((self.n_user, self.n_item))
        self.w0 = 0.
        self.w = np.zeros(self.p)
        self.V = np.random.normal(0., 0.1, (self.p, self.k))
        self.prev_w0 = float('inf')
        self.prev_w = np.array([])
        self.prev_V = np.array([])

    def _Base__predict(self, d):
        u_index = d['u_index']
        i_index = d['i_index']

        # create a sample vector and make prediction
        x = np.zeros(self.n_user + self.n_item)
        x[u_index] = x[self.n_user + i_index] = 1

        for ctx, dim in self.contexts:
            x = np.append(x, d[ctx])

        x_vec = np.array([x]).T  # p x 1
        interaction = np.sum(np.dot(self.V.T, x_vec) ** 2 - np.dot(self.V.T ** 2, x_vec ** 2)) / 2.
        return self.w0 + np.inner(self.w, x) + interaction

    def _Base__update(self, d, is_batch_train=False):
        u_index = d['u_index']
        i_index = d['i_index']

        # create a sample vector and make prediction
        x = np.zeros(self.n_user + self.n_item)
        x[u_index] = x[self.n_user + i_index] = 1

        for ctx, dim in self.contexts:
            x = np.append(x, d[ctx])

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
        # i_mat is (p, n_item) for all possible pairs of the user (d['u_index']) and all items
        i_mat = self.__create_i_mat(d)

        # Matrix A and B should be dense (numpy array; rather than scipy CSR matrix) because V is dense.
        A = safe_sparse_dot(self.V.T, i_mat)
        A = A ** 2

        sq_i_mat = i_mat.copy()
        sq_i_mat.data[:] = sq_i_mat.data ** 2
        B = safe_sparse_dot(self.V.T ** 2, sq_i_mat)

        interaction = np.sum(A - B, 0)
        interaction /= 2.  # (n_item,)

        pred = self.w0 + safe_sparse_dot(self.w, i_mat, dense_output=True) + interaction
        if self.is_positive_only:
            scores = np.abs(1. - (pred[:self.n_item] + np.sum(pred[self.n_item:])))
        else:
            scores = pred[:self.n_item] + np.sum(pred[self.n_item:])

        return self._Base__scores2recos(d['u_index'], scores, target_i_indices, at)

    def __create_i_mat(self, d):
        """Create a matrix which covers all possible <user, item> pairs for a given user index.

        Args:
            d (dict): A dictionary which has data of a sample.

        Returns:
            scipy csr_matrix (p, n_item): Column is a vector of <d['u_index'], item>.

        """
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
        ctx_head = 0
        for ctx, dim in self.contexts:
            row_ctx = np.array([])
            data_ctx = np.array([])
            for di in xrange(dim):
                row_ctx = np.append(row_ctx, np.ones(self.n_item) * (self.n_user + self.n_item + ctx_head + di))
                data_ctx = np.append(data_ctx, np.ones(self.n_item) * d[ctx][di])
            col_ctx = np.tile(np.arange(self.n_item), dim)

            row = np.append(row, row_ctx)
            col = np.append(col, col_ctx)
            data = np.append(data, data_ctx)

            ctx_head += dim

        return sp.csr_matrix((data, (row, col)), shape=(self.p, self.n_item))
