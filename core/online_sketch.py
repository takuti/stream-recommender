from logging import getLogger, StreamHandler, Formatter, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setFormatter(Formatter('[%(process)d] %(message)s'))
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

from .base import Base

import numpy as np
import numpy.linalg as ln
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.utils.extmath import safe_sparse_dot


class OnlineSketch(Base):

    """Inspired by: Streaming Anomaly Detection using Online Matrix Sketching
    """

    def __init__(self, contexts):

        self.contexts = contexts
        self.p = np.sum(list(contexts.values()))

        self.k = 40  # dimension of projected vectors
        self.ell = int(np.sqrt(self.k))

        self._Base__clear()

    # Since a batch training procedure is totally different from the matrix factorization techniques,
    # a public method "fit" is overridden.
    def fit(self, train_samples, test_samples, n_epoch=1, at=40):
        """Learn the "positve" sketched matrix using the first 30% positive samples to avoid cold-start.

        Args:
            train_samples (list of dict): Positive training samples (0-20%).
            test_sample (list of dict): Test samples (20-30%).
            at (int): Evaluation metric of this batch pre-training will be recall@{at}.
            n_epoch (int): Number of epochs for the batch training. (fixed by 1)

        """
        # make initial status for batch training
        for d in train_samples:
            self._Base__check(d)
            self.users[d['u_index']]['observed'].add(d['i_index'])

        # for batch evaluation, temporarily save new users info
        for d in test_samples:
            self._Base__check(d)

        Y0 = np.zeros((self.p, len(train_samples)))

        # 20%: update models
        for j, d in enumerate(train_samples):
            u = np.append(np.zeros(self.n_user), d['user'])
            u[d['u_index']] = 1.

            i = np.append(np.zeros(self.n_item), d['item'])
            i[d['i_index']] = 1.

            y = np.concatenate((u, d['others'], i))

            Y0[:, j] = y

        Y0 = np.dot(self.R, Y0)  # rabdom projection
        Y0 = preprocessing.normalize(Y0, norm='l2', axis=0)

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

        recall = self.batch_evaluate(test_samples, at)
        logger.debug('done: 20%% initial sketching with recall@%d = %f' % (at, recall[-1]))

        logger.debug('[' + ', '.join([str(r) for r in recall]) + ']')

        # for further incremental evaluation,
        # the model is incrementally updated by using the 10% samples
        for d in test_samples:
            self.users[d['u_index']]['observed'].add(d['i_index'])
            self._Base__update(d)

        logger.debug('done: additional learning for the 10% batch test samples')

    def _Base__clear(self):
        self.n_user = 0
        self.users = {}

        self.n_item = 0
        self.items = {}

        self.i_mat = sp.csr_matrix([])

        self.B = np.zeros((self.p, self.ell))

        # random projection matrix
        self.R = np.random.normal(0., 1 / self.k, (self.k, self.p))

    def _Base__check(self, d):

        u_index = d['u_index']
        if u_index not in self.users:
            self.users[u_index] = {'observed': set()}
            self.n_user += 1
            self.p += 1

            # projection matrix: insert a new column for new user ID
            col = np.random.normal(0., 1 / self.k, (self.k, 1))
            offset = self.n_user - 1
            self.R = np.concatenate((self.R[:, :offset], col, self.R[:, offset:]), axis=1)

        i_index = d['i_index']
        if i_index not in self.items:
            self.items[i_index] = {}
            self.n_item += 1
            self.p += 1

            i_vec = np.array([np.append(np.zeros(self.n_item), d['item'])]).T
            if self.i_mat.size == 0:
                self.i_mat = i_vec
            else:
                # item matrix: insert a new row for new item ID
                z = np.zeros((1, self.i_mat.shape[1]))
                self.i_mat = sp.csr_matrix(sp.vstack((self.i_mat[:(self.n_item - 1)],
                                                      z,
                                                      self.i_mat[(self.n_item - 1):])))
                self.i_mat = sp.csr_matrix(sp.hstack((self.i_mat, i_vec)))

            # projection matrix: insert a new column for new item ID
            col = np.random.normal(0., 1 / self.k, (self.k, 1))
            offset = self.n_user + self.contexts['user'] + self.contexts['others'] + self.n_item - 1
            self.R = np.concatenate((self.R[:, :offset], col, self.R[:, offset:]), axis=1)

    def _Base__update(self, d, is_batch_train=False):
        u = np.append(np.zeros(self.n_user), d['user'])
        u[d['u_index']] = 1.

        i = np.append(np.zeros(self.n_item), d['item'])
        i[d['i_index']] = 1.

        y = np.concatenate((u, d['others'], i))

        y = np.dot(self.R, y)  # random projection
        y = preprocessing.normalize(np.array([y]), norm='l2').flatten()

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
        # extract only target items
        i_mat = self.i_mat[:, target_i_indices]

        n_target = len(target_i_indices)

        # u_mat will be (n_user_context, n_item) for the target user
        u = np.concatenate((np.zeros(self.n_user), d['user'], d['others']))
        u[d['u_index']] = 1.
        u_vec = np.array([u]).T

        u_mat = sp.csr_matrix(np.repeat(u_vec, n_target, axis=1))

        # stack them into (p, n_item) matrix
        Y = sp.vstack((u_mat, i_mat))
        Y = safe_sparse_dot(self.R, Y)  # random projection -> dense output
        Y = sp.csr_matrix(preprocessing.normalize(Y, norm='l2', axis=0))

        X = np.identity(self.k) - np.dot(self.U, self.U.T)
        A = safe_sparse_dot(X, Y, dense_output=True)

        scores = ln.norm(A, axis=0, ord=2)

        return self._Base__scores2recos(scores, target_i_indices, at)
