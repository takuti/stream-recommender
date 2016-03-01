import time
import numpy as np

class IncrementalMF:
    def __init__(self, n_user, n_item, k=100, l2_reg=.01, learn_rate=.03):
        self.n_user = n_user
        self.n_item = n_item

        # parameter settings
        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg
        self.learn_rate = learn_rate

        self.__clear()

    def fit(self, train_samples):
        self.__clear()
        for d in train_samples:
            self.__update(d['u_index'], d['i_index'])

    def evaluate(self, test_samples, batch_flg, window_size=5000):
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
                recos = self.__recommend(u_index)

                # 2. Score the recommendation list given the true observed item i
                recall = 1 if (i_index in recos) else 0

                recalls[i % window_size] = recall
                s = float(sum(recalls))
                avg = s / window_size if (i + 1 >= window_size) else s / (i + 1)
                avgs.append(avg)
            else:
                avgs.append(avgs[-1])

            # 3.
            if batch_flg:
                # update the observed flag (w/o updating the model)
                self.observed[u_index, i_index] = 1
            else:
                # update the model with the observed event
                self.__update(u_index, i_index)

            ### stop timer
            total_time += (time.clock() - start)

        return avgs, total_time / float(len(test_samples))

    def __clear(self):
        self.observed = np.zeros((self.n_user, self.n_item))
        self.P = np.random.normal(0., 0.1, (self.n_user, self.k))
        self.Q = np.random.normal(0., 0.1, (self.n_item, self.k))

    def __update(self, u_index, i_index):
        self.observed[u_index, i_index] = 1

        u_vec = self.P[u_index]
        i_vec = self.Q[i_index]

        err = 1. - np.inner(u_vec, i_vec)

        next_u_vec = u_vec + 2. * self.learn_rate * (err * i_vec - self.l2_reg_u * u_vec)
        next_i_vec = i_vec + 2. * self.learn_rate * (err * u_vec - self.l2_reg_i * i_vec)
        self.P[u_index] = next_u_vec
        self.Q[i_index] = next_i_vec

    def __recommend(self, u_index, at=10):
        """
        Recommend Top-N items for the user u
        """

        recos = []
        pred = np.dot(np.array([self.P[u_index]]), self.Q.T)
        scores = np.abs(1. - pred.reshape(self.n_item))

        cnt = 0
        for i_index in np.argsort(scores):
            if self.observed[u_index, i_index] == 1: continue
            recos.append(i_index)
            cnt += 1
            if cnt == at: break

        return recos
