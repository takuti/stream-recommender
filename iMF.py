import numpy as np

class iMF:
    def __init__(self, n_user, n_item, k, l2_reg=.01, learn_rate=.01):

        self.n_user = n_user
        self.n_item = n_item
        self.known_users = np.array([])
        self.known_items = np.array([])

        # parameter settings
        self.k = k
        self.l2_reg_u = l2_reg
        self.l2_reg_i = l2_reg
        self.learn_rate = learn_rate

        self.A = np.random.normal(0., 0.1, (n_user, self.k))
        self.B = np.random.normal(0., 0.1, (n_item, self.k))

    def update(self, u_index, i_index):
        if u_index not in self.known_users: self.known_users = np.append(self.known_users, u_index)
        u_vec = self.A[u_index]

        if i_index not in self.known_items: self.known_items = np.append(self.known_items, i_index)
        i_vec = self.B[i_index]

        err = 1. - np.inner(u_vec, i_vec)
        next_u_vec = u_vec + 2. * self.learn_rate * (err * i_vec - self.l2_reg_u * u_vec)
        next_i_vec = i_vec + 2. * self.learn_rate * (err * u_vec - self.l2_reg_i * i_vec)
        self.A[u_index] = next_u_vec
        self.B[i_index] = next_i_vec

        #self.l2_reg_u = max(0., self.l2_reg_u + self.learn_rate * (err * self.learn_rate * (sum(next_u_vec) * sum(u_vec) - np.inner(next_u_vec, u_vec))))
        #self.l2_reg_i = max(0., self.l2_reg_i + self.learn_rate * (err * self.learn_rate * (sum(next_i_vec) * sum(i_vec) - np.inner(next_i_vec, i_vec))))

    def recommend(self, u_index, N, history_vec):
        """
        Recommend Top-N items for the user u
        """

        recos = []
        pred = np.dot(np.array([self.A[u_index]]), self.B.T)
        scores = np.abs(1. - pred.reshape(self.n_item))

        cnt = 0
        for i_index in np.argsort(scores):
            if history_vec[i_index] == 1: continue
            recos.append(i_index)
            cnt += 1
            if cnt == N: break

        return recos
