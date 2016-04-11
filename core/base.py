from abc import ABCMeta, abstractmethod

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

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

    def fit(self, train_samples, test_samples, at=10, n_epoch=1, is_monitoring=False):
        """Train a model using the first 30% positive samples to avoid cold-start.

        Evaluation of this batch training is done by using the next 20% positive samples.
        After the batch SGD training, the models are incrementally updated by using the 20% test samples.

        Args:
            train_samples (list of dict): Positive training samples (0-30%).
            test_sample (list of dict): Test samples (30-50%).
            at (int): Evaluation metric of this batch pre-training will be recall@{at}.
            n_epoch (int): Number of epochs for the batch training.

        """
        self.__clear()

        for epoch in range(n_epoch):
            # SGD requires us to shuffle samples in each iteration
            np.random.shuffle(train_samples)

            # 30%: update models
            for d in train_samples:
                u_index = d['u_index']
                i_index = d['i_index']
                self.observed[u_index, i_index] = 1

                self.__update(d, is_batch_train=True)

            # 20%: evaluate the current model
            if self.is_positive_only:
                recall = self.batch_evaluate_recall(test_samples, at)
                logger.debug('epoch %2d: recall@%d = %f' % (epoch + 1, at, recall))
            else:
                rmse = self.batch_evaluate_RMSE(test_samples)
                logger.debug('epoch %2d: RMSE = %f' % (epoch + 1, rmse))

        # for further incremental evaluation,
        # the model is incrementally updated by using the 20% samples
        if not is_monitoring:
            for d in test_samples:
                u_index = d['u_index']
                i_index = d['i_index']
                self.observed[u_index, i_index] = 1

                self.__update(d)

    def batch_evaluate_recall(self, test_samples, at):
        """Evaluate the current model by using the given test samples.

        Args:
            test_samples (list of dict): Current model is evaluated by these samples.
            at (int): Evaluation metric is recall@{at}.
                    For each sample,
                        top-{at} recommendation list has a true item -> TP++

        """
        n_tp = 0

        for i, d in enumerate(test_samples):
            # make recommendation for all items
            recos = self.__recommend(d, np.arange(self.n_item), at)

            # is a true sample in the top-{at} recommendation list?
            if d['i_index'] in recos:
                n_tp += 1

        return float(n_tp) / len(test_samples)

    def batch_evaluate_RMSE(self, test_samples):
        """Evaluate the current model by using the given test samples.

        Args:
            test_samples (list of dict): Current model is evaluated by these samples.

        """
        s = 0.

        # for each of test samples, make prediction and compute an error
        for i, d in enumerate(test_samples):
            s += ((d['y'] - self.__predict(d)) ** 2)

        return np.sqrt(s / len(test_samples))

    def evaluate(self, test_samples, window_size=500, at=10):
        """Iterate recommend/update procedure and compute incremental recall.

        Args:
            test_samples (list of dict): Positive test samples.
            at (int): Top-{at} items will be recommended in each iteration.

        Returns:
            float: incremental recalls@{at}.
            float: Avg. recommend+update time in second.

        """
        n_test = len(test_samples)
        recalls = np.zeros(n_test)

        window = np.zeros(window_size)
        sum_window = 0.

        # start timer
        start = time.clock()

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

    @abstractmethod
    def __clear(self):
        """Initialize model parameters.

        Observed flag array should be zero-cleared.

        """
        self.observed = np.zeros((self.n_user, self.n_item))
        pass

    @abstractmethod
    def __predict(self, d):
        """Make prediction for a sample 'd' using the current model parameters.

        Args:
            d (dict): A dictionary which has data of a sample.

        Returns:
            float: Predicted value.

        """
        return

    @abstractmethod
    def __update(self, d, is_batch_train):
        """Update model parameters based on d, a sample represented as a dictionary.

        Args:
            d (dict): A dictionary which has data of a sample.

        """
        u_index = d['u_index']
        i_index = d['i_index']
        self.observed[u_index, i_index] = 1
        pass

    @abstractmethod
    def __recommend(self, d, target_i_indices, at):
        """Recommend top-{at} items for a user represented as a dictionary d.

        First, scores are computed.
        Next, `self.__scores2recos()` is called to convert the scores into a recommendation list.

        Args:
            d (dict): A dictionary which has data of a sample.
            target_i_indices (numpy array; (# target items, )): Target items' indices. Only these items are considered as the recommendation candidates.
            at (int): Top-{at} items will be recommended.

        Returns:
            numpy array (at,): Recommendation list; top-{at} item indices.

        """
        return

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
        if not self.is_positive_only:
            # for explicit rating feedback, larger scores mean promising items
            sorted_indices = sorted_indices[::-1]

        for i_index in target_i_indices[sorted_indices]:
            # already observed <user, item> pair is NOT recommended
            if self.observed[u_index, i_index] == 1:
                continue

            recos = np.append(recos, i_index)
            if recos.size == at:
                break

        return recos.astype(int)
