from abc import ABCMeta, abstractmethod

import time
import numpy as np

class Base:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, n_user, n_item, **params):
        """Set/initialize parameters.

        """
        self.n_user = n_user
        self.n_item = n_item

        self.params = params

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
                recos = self.__recommend(d)

                # 2. Score the recommendation list given the true observed item i
                recall = 1 if (i_index in recos) else 0

                recalls[i % window_size] = recall
                s = float(sum(recalls))
                avg = s / window_size if (i + 1 >= window_size) else s / (i + 1)
                avgs.append(avg)
            else:
                avgs.append(avgs[-1])

            # update the model with the observed event
            self.__update(d)

            ### stop timer
            total_time += (time.clock() - start)

        return avgs, total_time / float(len(test_samples))

    @abstractmethod
    def __clear(self):
        """Clear model parameters.
        """
        self.observed = np.zeros((self.n_user, self.n_item))
        pass

    @abstractmethod
    def __update(self, d):
        """Update model parameters based on d, a sample represented as a dictionary.
        """
        u_index = d['u_index']
        i_index = d['i_index']
        self.observed[u_index, i_index] = 1
        pass

    @abstractmethod
    def __recommend(self, d, at):
        """Recommend top-{at} items for a user represented as a dictionary d.
        """
        return

    def __scores2recos(self, u_index, scores, at):
        """Get top-{at} recommendation list for a user u_index based on scores.

        """
        recos = np.array([])
        for i_index in np.argsort(scores):
            # already observed <user, item> pair is NOT recommended
            if self.observed[u_index, i_index] == 1: continue

            recos = np.append(recos, i_index)
            if recos.size == at: break
        return recos
