from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

from base import Base

import numpy as np
import numpy.linalg as ln
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot


class OnlineSketch(Base):

    """Inspired by: Streaming Anomaly Detection using Online Matrix Sketching
    """

    def __init__(self, n_user, n_item):
        """Set/initialize parameters.

        Args:
            n_user (int): Number of users.
            n_item (int): Number of items.

        """
        self.n_user = n_user
        self.n_item = n_item

        self.m = n_user + n_item
        self.ell = int(np.sqrt(self.m))

        self._Base__clear()

    # Since a batch training procedure is totally different from the matrix factorization techniques,
    # a public method "fit" is overridden.
    def fit(self, train_samples, test_samples, n_epoch=1, at=10, is_monitoring=False):
        """Learn the "positve" sketched matrix using the first 30% positive samples to avoid cold-start.

        Args:
            train_samples (list of dict): Positive training samples (0-30%).
            test_sample (list of dict): Test samples (30-50%).
            at (int): Evaluation metric of this batch pre-training will be recall@{at}.
            n_epoch (int): Number of epochs for the batch training. (fixed by 1)

        """
        Y0 = np.zeros((self.m, len(train_samples)))

        # 30%: update models
        for i, d in enumerate(train_samples):
            u_index = d['u_index']
            i_index = d['i_index']

            Y0[u_index, i] = Y0[self.n_user + i_index, i] = 1

        # initial ell orthogonal bases are computed by truncated SVD
        U, s, V = ln.svd(Y0, full_matrices=False)
        self.U = U[:, :self.ell]
        s = s[:self.ell]

        # shrink step in the Frequent Directions algorithm
        # (shrink singular values based on the squared smallest singular value)
        delta = s[-1] ** 2
        s = np.sqrt(s ** 2 - delta)

        # define initial sketched matrix B
        self.B = np.dot(self.U, np.diag(s))

        logger.debug('done: 30% initial sketching')

        # for further incremental evaluation,
        # the model is incrementally updated by using the 20% samples
        if not is_monitoring:
            for d in test_samples:
                u_index = d['u_index']
                i_index = d['i_index']
                self.observed[u_index, i_index] = 1

                self._Base__update(d)

            logger.debug('done: 20% additional learning')

    def _Base__clear(self):
        """Initialize model parameters.

        Observed flag array should be zero-cleared.

        """
        self.observed = np.zeros((self.n_user, self.n_item))

    def _Base__update(self, d, is_batch_train=False):
        # static baseline; w/o updating the model
        if not is_batch_train:
            return

        u_index = d['u_index']
        i_index = d['i_index']
        self.observed[u_index, i_index] = 1

        y = np.zeros((self.m, 1))
        y[u_index] = y[self.n_user + i_index] = 1.

        # combine current sketched matrix with input at time t
        # D: m-by-(ell+1) matrix
        D = np.hstack((self.B, y))

        U, s, V = ln.svd(D, full_matrices=False)

        # update ell orthogonal bases
        self.U = U[:, :self.ell]
        s = s[:self.ell]

        # shrink step in the Frequent Directions algorithm
        # (shrink singular values based on the squared smallest singular value)
        delta = s[-1] ** 2
        s = np.sqrt(s ** 2 - delta)

        self.B = np.dot(self.U, np.diag(s))

    def _Base__recommend(self, d, target_i_indices, at=10):
        u_index = d['u_index']

        # (n_user, n_item); user of d's row has 1s
        row_upper = np.ones(self.n_item) * u_index
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

        X = np.identity(self.m) - np.dot(self.U, self.U.T)
        Y = sp.csr_matrix((data, (row, col)), shape=(self.m, self.n_item))

        A = safe_sparse_dot(X, Y, dense_output=True)
        scores = ln.norm(A, axis=0, ord=2)

        return self._Base__scores2recos(u_index, scores, target_i_indices, at)
