import time
import numpy as np

class IncrementalBiasedMF:
    """
    Incremental Biased-MF as one specific case of Factorization Machines
    """

    def __init__(self, n_user, n_item, k=100, l2_reg_w0=.01, l2_reg_w=.01, l2_reg_V=.01, learn_rate=.03):

        self.n_user = n_user
        self.n_item = n_item

        # parameter settings
        self.k = k
        self.l2_reg_w0 = l2_reg_w0
        self.l2_reg_w = l2_reg_w
        self.l2_reg_V = l2_reg_V
        self.learn_rate = learn_rate

        self.p = n_user + n_item

        self.__clear()

    def fit(self, train_samples):
        self.__clear()
        for d in train_samples:
            self.__update(d['u_index'], d['i_index'])

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
                recos = self.__recommend(u_index)

                # 2. Score the recommendation list given the true observed item i
                recall = 1 if (i_index in recos) else 0

                recalls[i % window_size] = recall
                s = float(sum(recalls))
                avg = s / window_size if (i + 1 >= window_size) else s / (i + 1)
                avgs.append(avg)
            else:
                avgs.append(avgs[-1])

            # 3. update the model with the observed event
            self.__update(u_index, i_index)

            ### stop timer
            total_time += (time.clock() - start)

        return avgs, total_time / float(len(test_samples))

    def __clear(self):
        self.observed = np.zeros((self.n_user, self.n_item))
        self.w0 = 0.
        self.w = np.zeros(self.p)
        self.V = np.random.normal(0., 0.1, (self.p, self.k))
        self.prev_w0 = self.prev_w = float('inf')

    def __update(self, u_index, i_index):
        """
        Update the model parameters based on the given vector-value pair.
        """

        self.observed[u_index, i_index] = 1

        u = u_index
        i = self.n_user + i_index

        pred = np.inner(self.V[u], self.V[i]) + self.w0 + self.w[u] + self.w[i]
        err = 1. - pred

        # Updating regularization parameters
        if self.prev_w0 != float('inf'):
            self.l2_reg_w0 = max(0., self.l2_reg_w0 + 4. * self.learn_rate * (err * self.learn_rate * self.prev_w0))
            self.l2_reg_w = max(0., self.l2_reg_w + 4. * self.learn_rate * (err * self.learn_rate * (self.prev_w[u] + self.prev_w[i])))

        # Updating model parameters
        self.prev_w0 = self.w0
        self.w0 = self.w0 + 2. * self.learn_rate * (err * 1. - self.l2_reg_w0 * self.w0)

        # x_u and x_i are 1.0
        self.prev_w = np.empty_like(self.w)
        self.prev_w[:] = self.w
        self.w[u] = self.w[u] + 2. * self.learn_rate * (err * 1. - self.l2_reg_w * self.w[u])
        self.w[i] = self.w[i] + 2. * self.learn_rate * (err * 1. - self.l2_reg_w * self.w[i])

        next_u_vec = self.V[u] + 2. * self.learn_rate * (err * self.V[i] - self.l2_reg_V * self.V[u])
        next_i_vec = self.V[i] + 2. * self.learn_rate * (err * self.V[u] - self.l2_reg_V * self.V[i])
        self.V[u] = next_u_vec
        self.V[i] = next_i_vec

    def __recommend(self, u_index, at=10):

        recos = []

        i_offset = self.n_user

        pred = np.dot(np.array([self.V[u_index]]), self.V[i_offset:].T) + self.w0 + self.w[u_index] + np.array([self.w[i_offset:]])
        scores = np.abs(1. - pred.reshape(self.n_item))

        cnt = 0
        for i_index in np.argsort(scores):
            if self.observed[u_index, i_index] == 1: continue
            recos.append(i_index)
            cnt += 1
            if cnt == at: break

        return recos

