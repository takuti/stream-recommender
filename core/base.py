from abc import ABCMeta, abstractmethod

import time
import numpy as np

class Base:
    """Base class for experimentation of the incremental models with positive-only feedback.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, n_user, n_item, **params):
        """Set/initialize parameters.

        Args:
            n_user (int): Number of users.
            n_item (int): Number of items.

        """
        self.n_user = n_user
        self.n_item = n_item

        # set parameters
        self.params = params

        # initialize models
        self.__clear()

    def fit(self, train_samples):
        """Train a model using the first 20% positive samples to avoid cold-start.

        Args:
            train_samples (list of dict): Positive training samples.

        """
        self.__clear()
        for d in train_samples:
            self.__update(d)

    def evaluate(self, test_samples, window_size=5000, at=10):
        """Iterate recommend/update procedure and compute Simple Moving Averages (SMAs).

        Args:
            test_samples (list of dict): Positive test samples.
            window_size (int): For SMA.
            at (int): Top-{at} items will be recommended in each iteration.

        Returns:
            numpy array (n_test,): SMAs corresponding to iteration over the test samples.
            float: Avg. recommend+update time in second.

        """
        n_test = len(test_samples)

        window = np.zeros(window_size)
        sum_window = 0.

        avgs = np.zeros(n_test)
        latest_avg = 0.

        ### start timer
        start = time.clock()

        for i, d in enumerate(test_samples):
            u_index = d['u_index']
            i_index = d['i_index']

            # Step 1: if u is a known user, recommend items using current model
            if 1 in self.observed[u_index, :]:
                recos = self.__recommend(d, at)

                # score the recommendation list based on the true observed item
                wi = i % window_size

                old_recall = window[wi]
                new_recall = 1. if (i_index in recos) else 0.
                window[wi] = new_recall

                sum_window = sum_window - old_recall + new_recall
                latest_avg = sum_window / min(i+1, window_size)

            # save the latest average
            # if u is unobserved user, avg of this step will be same as the previous avg
            avgs[i] = latest_avg

            # Step 2: update the model with the observed event
            self.__update(d)

        ### stop timer
        avg_time = (time.clock() - start) / n_test

        return avgs, avg_time

    @abstractmethod
    def __clear(self):
        """Initialize model parameters.

        Observed flag array should be zero-cleared.

        """
        self.observed = np.zeros((self.n_user, self.n_item))
        pass

    @abstractmethod
    def __update(self, d):
        """Update model parameters based on d, a sample represented as a dictionary.

        Args:
            d (dict): A dictionary which has data of a sample.

        """
        u_index = d['u_index']
        i_index = d['i_index']
        self.observed[u_index, i_index] = 1
        pass

    @abstractmethod
    def __recommend(self, d, at):
        """Recommend top-{at} items for a user represented as a dictionary d.

        First, scores are computed.
        Next, `self.__scores2recos()` is called to convert the scores into a recommendation list.

        Args:
            d (dict): A dictionary which has data of a sample.
            at (int): Top-{at} items will be recommended.

        Returns:
            numpy array (at,): Recommendation list; top-{at} item indices.

        """
        return

    def __scores2recos(self, u_index, scores, at):
        """Get top-{at} recommendation list for a user u_index based on scores.

        Args:
            u_index (int): Target user's index.
            scores (numpy array; (n_item,)): Scores for every items. Smaller score indicates a promising item.
            at (int): Top-{at} items will be recommended.

        Returns:
            numpy array (at,): Recommendation list; top-{at} item indices.

        """
        recos = np.array([])
        for i_index in np.argsort(scores):
            # already observed <user, item> pair is NOT recommended
            if self.observed[u_index, i_index] == 1: continue

            recos = np.append(recos, i_index)
            if recos.size == at: break
        return recos
