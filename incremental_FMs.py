import numpy as np

class IncrementalFMs:
    """
    Incremental Biased-MF as one specific case of Factorization Machines
    """

    def __init__(self, n_user, n_item, k, l2_reg_w0=.01, l2_reg_w=.01, l2_reg_V=.01, learn_rate=.01):

        self.n_user = n_user
        self.n_item = n_item
        self.known_users = np.array([])
        self.known_items = np.array([])

        # parameter settings
        self.k = k
        self.l2_reg_w0 = l2_reg_w0
        self.l2_reg_w = l2_reg_w
        #self.l2_reg_V = np.ones(self.k) * l2_reg_V # each of the factorization dimensionality
        self.l2_reg_V = l2_reg_V
        self.learn_rate = learn_rate

        self.p = n_user + n_item

        # initialize the model parameters
        self.w0 = 0.
        self.w = np.zeros(self.p)
        self.V = np.random.normal(0., 0.1, (self.p, self.k))

    def update(self, u_index, i_index, prev_w0=None, prev_w=None):
        """
        Update the model parameters based on the given vector-value pair.
        """

        u = u_index
        i = self.n_user + i_index
        x = np.zeros(self.p)
        x[u] = x[i] = 1.

        if u_index not in self.known_users: self.known_users = np.append(self.known_users, u_index)
        if i_index not in self.known_items: self.known_items = np.append(self.known_items, i_index)

        #interaction = float(np.sum(np.dot(self.V.T, np.array([x]).T) ** 2 - np.dot(self.V.T ** 2, np.array([x]).T ** 2)) / 2.)
        #pred = self.w0 + np.inner(self.w, x) + interaction
        pred = np.inner(self.V[u], self.V[i]) + self.w0 + self.w[u] + self.w[i]

        err = 1. - pred

        # Updating regularization parameters
        if prev_w0 != None and prev_w != None:
            self.l2_reg_w0 = max(0., self.l2_reg_w0 + 4. * self.learn_rate * (err * self.learn_rate * prev_w0))
            self.l2_reg_w = max(0., self.l2_reg_w + 4. * self.learn_rate * (err * self.learn_rate * np.inner(x, prev_w)))

        # Updating model parameters
        prev_w0 = self.w0
        self.w0 = self.w0 + 2. * self.learn_rate * (err * 1. - self.l2_reg_w0 * self.w0)

        # x_u and x_i are 1.0
        prev_w = np.empty_like(self.w)
        prev_w[:] = self.w

        prev_V = np.empty_like(self.V)
        prev_V[:] = self.V

        for pi in xrange(self.p):
            if x[pi] == 0.: continue

            self.w[pi] = prev_w[pi] + 2. * self.learn_rate * (err * x[pi] - self.l2_reg_w * prev_w[pi])

            g = err * x[pi] * (np.dot(np.array([x]), prev_V) - x[pi] * prev_V[pi])
            self.V[pi] = prev_V[pi] + 2. * self.learn_rate * (g - self.l2_reg_V * prev_V[pi])

            """
            for ki in xrange(self.k):
                g = err * x[pi] * (np.inner(x, prev_V[:, ki]) - x[pi] * prev_V[pi, ki])
                self.V[pi, ki] = prev_V[pi, ki] + 2. * self.learn_rate * (g - self.l2_reg_V * prev_V[pi, ki])
            """

        """
        next_u_vec = self.V[u] + 2. * self.learn_rate * (err * self.V[i] - self.l2_reg_V * self.V[u])
        next_i_vec = self.V[i] + 2. * self.learn_rate * (err * self.V[u] - self.l2_reg_V * self.V[i])
        self.V[u] = next_u_vec
        self.V[i] = next_i_vec
        """

        return prev_w0, prev_w

    def recommend(self, u_index, N, history_vec):
        recos = []

        i_offset = self.n_user

        pred = np.dot(np.array([self.V[u_index]]), self.V[i_offset:].T) + self.w0 + self.w[u_index] + np.array([self.w[i_offset:]])
        scores = np.abs(1. - pred.reshape(self.n_item))

        cnt = 0
        for i_index in np.argsort(scores):
            if history_vec[i_index] == 1: continue
            recos.append(i_index)
            cnt += 1
            if cnt == N: break

        return recos

