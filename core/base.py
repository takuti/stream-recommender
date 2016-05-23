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
    def __init__(self, n_item, **params):
        """Set/initialize parameters.

        Args:
            n_item (int): Number of pre-defined items.

        """
        self.n_item = n_item

        # set parameters
        self.params = params

        # initialize models and user/item information
        self.__clear()

    def fit(self, train_samples, test_samples, at=10, n_epoch=1):
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

        # make initial status for batch training
        for d in train_samples:
            self.__check(d)
            self.users[d['u_index']]['observed'].add(d['i_index'])

        # for batch evaluation, temporarily save new users info
        for d in test_samples:
            self.__check(d)

        self.batch_update(train_samples, test_samples, at, n_epoch)

        # batch test samples are considered as a new observations;
        # the model is incrementally updated based on them before the incremental evaluation step
        for d in test_samples:
            self.users[d['u_index']]['observed'].add(d['i_index'])
            self.__update(d)

    def batch_update(self, train_samples, test_samples, at, n_epoch):
        """Batch update called by the fitting method.

        Args:
            train_samples (list of dict): Positive training samples (0-20%).
            test_sample (list of dict): Test samples (20-30%).
            at (int): Evaluation metric of this batch pre-training will be recall@{at}.
            n_epoch (int): Number of epochs for the batch training.

        """
        for epoch in range(n_epoch):
            # SGD requires us to shuffle samples in each iteration
            np.random.shuffle(train_samples)

            # 20%: update models
            for d in train_samples:
                self.__update(d, is_batch_train=True)

            # 10%: evaluate the current model
            recall = self.batch_evaluate(test_samples, at)
            logger.debug('epoch %2d: recall@%d = %f' % (epoch + 1, at, recall))

    def batch_evaluate(self, test_samples, at):
        """Evaluate the current model by using the given test samples.

        Args:
            test_samples (list of dict): Current model is evaluated by these samples.
            at (int): Evaluation metric is recall@{at}.
                    For each sample,
                        top-{at} recommendation list has a true item -> TP++

        """
        n_tp = 0

        all_items = set(range(self.n_item))
        for i, d in enumerate(test_samples):
            # make recommendation for all unobserved items
            unobserved = all_items - self.users[d['u_index']]['observed']
            unobserved.add(d['i_index'])  # true item itself must be in the recommendation candidates
            target_i_indices = np.asarray(list(unobserved))
            recos = self.__recommend(d, target_i_indices, at)

            # is a true sample in the top-{at} recommendation list?
            if d['i_index'] in recos:
                n_tp += 1

        return float(n_tp) / len(test_samples)

    def evaluate(self, test_samples, window_size=5000, at=10):
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
            self.__check(d)

            u_index = d['u_index']
            i_index = d['i_index']

            # unobserved items
            # * item i interacted by user u must be in the recommendation candidate
            unobserved = set(range(self.n_item)) - self.users[u_index]['observed']
            unobserved.add(i_index)
            target_i_indices = np.asarray(list(unobserved))

            # make top-{at} recommendation for the 1001 items
            recos = self.__recommend(d, target_i_indices, at)

            # increment a hit counter if i_index is in the top-{at} recommendation list
            # i.e. score the recommendation list based on the true observed item
            wi = i % window_size

            old_recall = window[wi]
            new_recall = 1. if (i_index in recos) else 0.
            window[wi] = new_recall

            sum_window = sum_window - old_recall + new_recall
            recalls[i] = sum_window / min(i + 1, window_size)

            # Step 2: update the model with the observed event
            self.users[u_index]['observed'].add(i_index)
            self.__update(d)

        # stop timer
        avg_time = (time.clock() - start) / n_test

        return recalls, avg_time

    @abstractmethod
    def __clear(self):
        """Initialize model parameters and user/item info.

        """
        self.n_user = 0
        self.users = {}
        pass

    @abstractmethod
    def __check(self, d):
        """Check if user/item is new.

        For new users/items, append their information into the dictionaries.

        """
        u_index = d['u_index']

        if u_index not in self.users:
            self.users[u_index] = {'observed': set()}
            self.n_user += 1

        pass

    @abstractmethod
    def __update(self, d, is_batch_train):
        """Update model parameters based on d, a sample represented as a dictionary.

        Args:
            d (dict): A dictionary which has data of a sample.

        """
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

    def __scores2recos(self, scores, target_i_indices, at):
        """Get top-{at} recommendation list for a user u_index based on scores.

        Args:
            scores (numpy array; (n_target_items,)):
                Scores for the target items. Smaller score indicates a promising item.
            target_i_indices (numpy array; (# target items, )): Target items' indices. Only these items are considered as the recommendation candidates.
            at (int): Top-{at} items will be recommended.

        Returns:
            numpy array (at,): Recommendation list; top-{at} item indices.

        """

        sorted_indices = np.argsort(scores)
        return target_i_indices[sorted_indices][:at]
