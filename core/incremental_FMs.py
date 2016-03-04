import time
import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot

class IncrementalFMs:
    """
    Incremental Factorization Machines
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

        self.__clear()

    def fit(self, train_samples):
        self.__clear()
        for d in train_samples:
            self.__update(d)

    def evaluate(self, test_samples, window_size=5000):
        total_time = 0.
        recalls = [0] * window_size
        avgs = []

        for i, d in enumerate(test_samples):
            u_index = d['u_index']
            i_index = d['i_index']

            ### start timer
            start = time.clock()

            # 1.
            if 1 in self.observed[u_index, :]:
                # If u is a known user, use the current model to recommend N items,
                recos = self.__recommend(u_index, self.__create_i_mat(d))

                # 2. Score the recommendation list given the true observed item i
                recall = 1 if (i_index in recos) else 0

                recalls[i % window_size] = recall
                s = float(sum(recalls))
                avg = s / window_size if (i + 1 >= window_size) else s / (i + 1)
                avgs.append(avg)
            else:
                avgs.append(avgs[-1])

            # 3. update the model with the observed event
            self.__update(d)

            ### stop timer
            total_time += (time.clock() - start)

        return avgs, total_time / float(len(test_samples))

    def __clear(self):
        self.observed = np.zeros((self.n_user, self.n_item))
        self.w0 = 0.
        self.w = np.zeros(self.p)
        self.V = np.random.normal(0., 0.1, (self.p, self.k))
        self.prev_w0 = self.prev_w = float('inf')

    def __update(self, d):
        """
        Update the model parameters based on the given vector-value pair.
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

        #u = u_index
        #i = self.n_user + i_index
        #pred = np.inner(self.V[u], self.V[i]) + self.w0 + self.w[u] + self.w[i]
        err = 1. - pred

        # Updating regularization parameters
        if self.prev_w0 != float('inf'):
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

    def __recommend(self, u_index, i_mat, at=10):
        recos = []

        i_offset = self.n_user

        # i_mat is (p, n_item) for all possible pairs of the user (u_index) and all items
        A = safe_sparse_dot(self.V.T, i_mat)
        if sp.issparse(A):
            A.data[:] = A.data ** 2
        else:
            A = A ** 2

        sq_i_mat = i_mat.copy()
        sq_i_mat.data[:] = sq_i_mat.data ** 2
        B = safe_sparse_dot(self.V.T ** 2, sq_i_mat)

        interaction = np.sum(A - B, 0) if (not sp.issparse(A) and not sp.issparse(B)) else sp.csr_matrix.sum(A - B, 0)
        interaction /= 2. # (n_item,)

        pred = self.w0 + safe_sparse_dot(self.w, i_mat, dense_output=True) + interaction

        scores = np.abs(1. - (pred[:self.n_item] + np.sum(pred[self.n_item:])))

        cnt = 0
        for i_index in np.argsort(scores):
            if self.observed[u_index, i_index] == 1: continue
            recos.append(i_index)
            cnt += 1
            if cnt == at: break

        return recos

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
