from abc import ABCMeta, abstractmethod
from .base import Base

import numpy as np
import numpy.linalg as ln
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.utils.extmath import safe_sparse_dot


class BaseProjection:

    """Base class for projection of context-aware matrix.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, k, p):
        """Initialize projection matrices.

        Args:
            k (int): Number of reduced dimensions (i.e. rows of projection mat.).
            p (int): Number of input dimensions (i.e. cols of projection mat.).
                p will be increased in the future due to new user/item/context insersion.

        """
        pass

    @abstractmethod
    def insert_proj_col(self, offset):
        """Insert a new column for a projection matrix.

        Args:
            offset (int): Index of the inserted column.

        """
        pass

    @abstractmethod
    def reduce(self, Y):
        """Make projection for an input matrix.

        Args:
            Y (numpy array; (p, n)): Input p-by-n matrix projected to a k-by-n matrix.

        Returns:
            numpy array; (k, n): Projected matrix.

        """
        return


class Raw(BaseProjection):

    def __init__(self, p):
        self.I = np.identity(p)

    def insert_proj_col(self, offset):
        pass

    def reduce(self, Y):
        return safe_sparse_dot(self.I, Y)


class RandomProjection(BaseProjection):

    def __init__(self, k, p):
        self.k = k
        self.R = sp.csr_matrix(self.__create_proj_mat((k, p)))

    def insert_proj_col(self, offset):
        col = self.__create_proj_mat((self.k, 1))
        R = self.R.toarray()
        self.R = sp.csr_matrix(np.concatenate((R[:, :offset], col, R[:, offset:]), axis=1))

    def reduce(self, Y):
        return safe_sparse_dot(self.R, Y)

    def __create_proj_mat(self, size):
        """Create a random projection matrix

        [1] D. Achlioptas. Database-friendly random projections: Johnson-Lindenstrauss with binary coins.
        [2] P. Li, et al. Very sparse random projections.

        http://scikit-learn.org/stable/modules/random_projection.html#sparse-random-projection
        """

        # [1]
        # return np.random.choice([-np.sqrt(3), 0, np.sqrt(3)], size=size, p=[1 / 6, 2 / 3, 1 / 6])

        # [2]
        s = 1 / 0.2
        return np.random.choice([-np.sqrt(s / self.k), 0, np.sqrt(s / self.k)],
                                size=size,
                                p=[1 / (2 * s), 1 - 1 / s, 1 / (2 * s)])


class RandomMaclaurinProjection(BaseProjection):

    def __init__(self, k, p):
        self.k = k

        self.W1 = np.random.choice([1, -1], size=(k, p))
        self.W2 = np.random.choice([1, -1], size=(k, p))

    def insert_proj_col(self, offset):
        col = np.random.choice([1, -1], size=(self.k, 1))
        self.W1 = np.concatenate((self.W1[:, :offset], col, self.W1[:, offset:]), axis=1)

        col = np.random.choice([1, -1], size=(self.k, 1))
        self.W2 = np.concatenate((self.W2[:, :offset], col, self.W2[:, offset:]), axis=1)

    def reduce(self, Y):
        return safe_sparse_dot(self.W1, Y) * safe_sparse_dot(self.W2, Y) / np.sqrt(self.k)


class TensorSketchProjection(BaseProjection):

    def __init__(self, k, p):
        self.k = k

        self.h1 = np.random.choice(range(k), size=p)
        self.h2 = np.random.choice(range(k), size=p)
        self.h1_indices = [np.where(self.h1 == j)[0] for j in range(k)]
        self.h2_indices = [np.where(self.h2 == j)[0] for j in range(k)]
        self.s1 = np.random.choice([1, -1], size=p)
        self.s2 = np.random.choice([1, -1], size=p)

    def insert_proj_col(self, offset):
        self.h1 = np.concatenate((self.h1[:offset],
                                  np.random.choice(range(self.k), (1, )),
                                  self.h1[offset:]))
        self.h2 = np.concatenate((self.h2[:offset],
                                  np.random.choice(range(self.k), (1, )),
                                  self.h2[offset:]))
        self.h1_indices = [np.where(self.h1 == j)[0] for j in range(self.k)]
        self.h2_indices = [np.where(self.h2 == j)[0] for j in range(self.k)]
        self.s1 = np.concatenate((self.s1[:offset], np.random.choice([1, -1], (1, )), self.s1[offset:]))
        self.s2 = np.concatenate((self.s2[:offset], np.random.choice([1, -1], (1, )), self.s2[offset:]))

    def reduce(self, Y):
        if sp.isspmatrix(Y):
            Y = Y.toarray()

        sketch1, sketch2 = self.__sketch(Y)

        return np.real(np.fft.ifft(np.fft.fft(sketch1, axis=0) * np.fft.fft(sketch2, axis=0), axis=0))

    def __sketch(self, X):
        sketch1 = np.array([np.sum(np.array([self.s1[idx]]).T * X[idx], axis=0) for idx in self.h1_indices])
        sketch2 = np.array([np.sum(np.array([self.s2[idx]]).T * X[idx], axis=0) for idx in self.h2_indices])

        return sketch1, sketch2


class OnlineSketch(Base):

    """Inspired by: Streaming Anomaly Detection using Online Matrix Sketching
    """

    def __init__(self, contexts, k=40):

        self.contexts = contexts
        self.p = np.sum(list(contexts.values()))

        self.k = self.p  # dimension of projected vectors
        self.ell = int(np.sqrt(self.k))

        self._Base__clear()

    def _Base__clear(self):
        self.n_user = 0
        self.users = {}

        self.n_item = 0
        self.items = {}

        self.i_mat = sp.csr_matrix([])

        self.B = np.zeros((self.k, self.ell))

        # initialize projection instance
        self.proj = Raw(self.p)

    def _Base__check(self, d):

        u_index = d['u_index']
        if u_index not in self.users:
            self.users[u_index] = {'observed': set()}
            self.n_user += 1

        i_index = d['i_index']
        if i_index not in self.items:
            self.items[i_index] = {}
            self.n_item += 1

            i_vec = sp.csr_matrix(np.array([d['item']]).T)
            if self.i_mat.size == 0:
                self.i_mat = i_vec
            else:
                self.i_mat = sp.csr_matrix(sp.hstack((self.i_mat, i_vec)))

    def _Base__update(self, d, is_batch_train=False):
        y = np.concatenate((d['user'], d['others'], d['item']))
        y = self.proj.reduce(np.array([y]).T)
        y = np.ravel(preprocessing.normalize(y, norm='l2', axis=0))

        # combine current sketched matrix with input at time t
        zero_cols = np.where(np.isclose(self.B, 0).all(0) == 1)[0]
        j = zero_cols[0] if zero_cols.size != 0 else self.ell - 1  # left-most all-zero column in B
        self.B[:, j] = y

        U, s, V = ln.svd(self.B, full_matrices=False)

        # update ell orthogonal bases
        self.U = U[:, :self.ell]
        s = s[:self.ell]

        # shrink step in the Frequent Directions algorithm
        # (shrink singular values based on the squared smallest singular value)
        delta = s[-1] ** 2
        s = np.sqrt(s ** 2 - delta)

        self.B = np.dot(self.U, np.diag(s))

    def _Base__recommend(self, d, target_i_indices):
        # i_mat is (n_item_context, n_item) for all possible items
        # extract only target items
        i_mat = self.i_mat[:, target_i_indices]

        n_target = len(target_i_indices)

        # u_mat will be (n_user_context, n_item) for the target user
        u = np.concatenate((d['user'], d['others']))
        u_vec = np.array([u]).T

        u_mat = sp.csr_matrix(np.repeat(u_vec, n_target, axis=1))

        # stack them into (p, n_item) matrix
        Y = sp.vstack((u_mat, i_mat))
        Y = self.proj.reduce(Y)
        Y = sp.csr_matrix(preprocessing.normalize(Y, norm='l2', axis=0))

        X = np.identity(self.k) - np.dot(self.U, self.U.T)
        A = safe_sparse_dot(X, Y, dense_output=True)

        scores = ln.norm(A, axis=0, ord=2)

        return self._Base__scores2recos(scores, target_i_indices)
