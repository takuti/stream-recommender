from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

import time
import numpy as np
import numpy.linalg as ln
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot


class OnlineSketch:

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

        self.__clear()

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

        print '[log] initial sketch'

        # for further incremental evaluation,
        # the model is incrementally updated by using the 20% samples
        if not is_monitoring:
            for d in test_samples:
                u_index = d['u_index']
                i_index = d['i_index']
                self.observed[u_index, i_index] = 1

                self.__update(d)

        print '[log] incremental update 20%'

    def evaluate(self, test_samples, window_size=500, at=10):
        """Iterate recommend/update procedure and compute incremental recall.

        Args:
            test_samples (list of dict): Positive test samples.
            at (int): Top-{at} items will be recommended in each iteration.

        Returns:
            float: incremental recalls@{at}.
            float: Avg. recommend+update time in second.

        """
        test_samples = test_samples[:5000]
        n_test = len(test_samples)
        recalls = np.zeros(n_test)

        window = np.zeros(window_size)
        sum_window = 0.

        # start timer
        start = time.clock()

        print '[log] start evaluate'
        for i, d in enumerate(test_samples):
            u_index = d['u_index']
            i_index = d['i_index']

            self.observed[u_index, i_index] = 0

            # 1000 further unobserved items + item i interacted by user u
            unobserved_i_indices = np.where(self.observed[u_index, :] == 0)[0]
            n_unobserved = unobserved_i_indices.size
            target_i_indices = np.random.choice(unobserved_i_indices, min(n_unobserved, 1000), replace=False)

            # make top-{at} recommendation for the 1001 items
            recos = self.__recommend(d, target_i_indices, at)

            self.observed[u_index, i_index] = 1

            # increment a hit counter if i_index is in the top-{at} recommendation list
            # i.e. score the recommendation list based on the true observed item
            wi = i % window_size

            old_recall = window[wi]
            new_recall = 1. if (i_index in recos) else 0.
            window[wi] = new_recall

            sum_window = sum_window - old_recall + new_recall
            recalls[i] = sum_window / min(i + 1, window_size)

            # Step 2: update the model with the observed event
            self.__update(d)

        # stop timer
        avg_time = (time.clock() - start) / n_test

        return recalls, avg_time

    def __clear(self):
        """Initialize model parameters.

        Observed flag array should be zero-cleared.

        """
        self.observed = np.zeros((self.n_user, self.n_item))

    def __update(self, d, is_batch_train=False):
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

    def __recommend(self, d, target_i_indices, at=10):
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

        return self.__scores2recos(u_index, scores, target_i_indices, at)

    def __scores2recos(self, u_index, scores, target_i_indices, at):
        """Get top-{at} recommendation list for a user u_index based on scores.

        Args:
            u_index (int): Target user's index.
            scores (numpy array; (n_item,)): Scores for every items. Smaller score indicates a promising item.
            target_i_indices (numpy array; (# target items, )): Target items' indices. Only these items are considered as the recommendation candidates.
            at (int): Top-{at} items will be recommended.

        Returns:
            numpy array (at,): Recommendation list; top-{at} item indices.

        """
        recos = np.array([])
        target_scores = scores[target_i_indices]

        sorted_indices = np.argsort(target_scores)

        for i_index in target_i_indices[sorted_indices]:
            # already observed <user, item> pair is NOT recommended
            if self.observed[u_index, i_index] == 1:
                continue

            recos = np.append(recos, i_index)
            if recos.size == at:
                break

        return recos.astype(int)
