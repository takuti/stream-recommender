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

    def __init__(self, n_item, samples, contexts):
        self.n_item = n_item

        self.contexts = contexts
        self.p = np.sum(contexts.values())
        self.ell = int(np.sqrt(self.p))

        # create item matrices which has contexts of each item in rows
        self.i_mat = self.__create_i_mat(samples)

        self._Base__clear()

    # Since a batch training procedure is totally different from the matrix factorization techniques,
    # a public method "fit" is overridden.
    def fit(self, train_samples, test_samples, n_epoch=1, at=10):
        """Learn the "positve" sketched matrix using the first 30% positive samples to avoid cold-start.

        Args:
            train_samples (list of dict): Positive training samples (0-20%).
            test_sample (list of dict): Test samples (20-30%).
            at (int): Evaluation metric of this batch pre-training will be recall@{at}.
            n_epoch (int): Number of epochs for the batch training. (fixed by 1)

        """
        Y0 = np.zeros((self.p, len(train_samples)))

        # 20%: update models
        for i, d in enumerate(train_samples):
            self._Base__check(d)
            self.users[d['u_index']]['observed'].add(d['i_index'])

            y = d['user']
            y = np.append(y, d['item'])
            y = np.append(y, d['dt'])

            Y0[:, i] = y

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

        logger.debug('done: 20% initial sketching')

        # for further incremental evaluation,
        # the model is incrementally updated by using the 10% samples
        for d in test_samples:
            self._Base__check(d)
            self.users[d['u_index']]['observed'].add(d['i_index'])

            self._Base__update(d)

        logger.debug('done: additional learning for the 10% batch test samples')

    def _Base__clear(self):
        self.n_user = 0
        self.users = {}

        self.B = np.random.normal(0., 0.1, (self.p, self.ell))

    def _Base__check(self, d):
        u_index = d['u_index']

        if u_index not in self.users:
            self.users[u_index] = {'observed': set()}
            self.n_user += 1

    def _Base__update(self, d, is_batch_train=False):
        y = d['user']
        y = np.append(y, d['item'])
        y = np.append(y, d['dt'])

        # combine current sketched matrix with input at time t
        self.B[:, (self.ell - 1)] = y

        U, s, V = ln.svd(self.B, full_matrices=False)

        # update ell orthogonal bases
        self.U = U[:, :self.ell]
        s = s[:self.ell]

        # shrink step in the Frequent Directions algorithm
        # (shrink singular values based on the squared smallest singular value)
        delta = s[-1] ** 2
        s = np.sqrt(s ** 2 - delta)

        self.B = np.dot(self.U, np.diag(s))

    def _Base__recommend(self, d, target_i_indices, at=10):
        # i_mat is (n_item_context, n_item) for all possible items

        # u_mat will be (n_user_context, n_item) for the target user
        u_mat = np.repeat(np.array([d['user']]).T, self.n_item, axis=1)

        # dt_mat will be (1, n_item) for the target user
        dt_mat = np.repeat(d['dt'], self.n_item)

        # stack them into (p, n_item) matrix
        Y = sp.csr_matrix(np.vstack((u_mat, self.i_mat, dt_mat))[:, target_i_indices])

        X = np.identity(self.p) - np.dot(self.U, self.U.T)
        A = safe_sparse_dot(X, Y, dense_output=True)

        scores = ln.norm(A, axis=0, ord=2)

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
            observed[i_index] = 1

        return i_mat.T
