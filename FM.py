import numpy as np

class FM:
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

        if u_index not in self.known_users: self.known_users = np.append(self.known_users, u_index)
        u_vec = self.V[u]

        if i_index not in self.known_items: self.known_items = np.append(self.known_items, i_index)
        i_vec = self.V[i]

        pred = np.inner(self.V[u], self.V[i]) + self.w0 + self.w[u] + self.w[i]
        err = 1. - pred

        # Updating regularization parameters
        if prev_w0 != None and prev_w != None:
            self.l2_reg_w0 = max(0., self.l2_reg_w0 + 4. * self.learn_rate * (err * self.learn_rate * prev_w0))
            self.l2_reg_w = max(0., self.l2_reg_w + 4. * self.learn_rate * (err * self.learn_rate * (prev_w[u] + prev_w[i])))

        # Updating model parameters
        prev_w0 = self.w0
        self.w0 = self.w0 + 2. * self.learn_rate * (err * 1. - self.l2_reg_w0 * self.w0)

        # x_u and x_i are 1.0
        prev_w = self.w
        self.w[u] = self.w[u] + 2. * self.learn_rate * (err * 1. - self.l2_reg_w * self.w[u])
        self.w[i] = self.w[i] + 2. * self.learn_rate * (err * 1. - self.l2_reg_w * self.w[i])

        #s = np.sum(self.V, axis=0)
        #prev_V = np.ones((self.p, self.k)) * self.V
        self.V[u] = u_vec + 2. * self.learn_rate * (err * i_vec - self.l2_reg_V * u_vec)
        self.V[i] = i_vec + 2. * self.learn_rate * (err * u_vec - self.l2_reg_V * i_vec)

        return prev_w0, prev_w
        # Updating regularization parameters (use both 't' and 't+1')
        #self.l2_reg_V = np.maximum(0., self.l2_reg_V + self.learn_rate * (err * self.learn_rate * (np.sum(self.V, axis=0) * prev_sum - np.sum(self.V * prev_V, axis=0))))

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

