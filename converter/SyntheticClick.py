# conding: utf-8

import numpy as np


class SyntheticClickConverter:

    """Dataset will be generated as:

    $ julia clickgenerator.jl > click.tsv
    """

    def __init__(self):
        # contexts in this dataset
        # others: none
        # user: 1 normalized age, 1 sex, 50 for living state
        # item: none
        self.contexts = {'others': 1, 'user': 52, 'item': 1}

    def convert(self):
        """Create a list of samples and count number of users/items.

        """

        clicks = []
        with open('converter/click.tsv') as f:
            clicks = list(map(lambda l: list(map(float, l.rstrip().split('\t'))), f.readlines()))

        self.samples = []

        u_index = 0  # each sample indicates different visitors
        n_geo = 50  # 50 states in US

        ad_ids = []

        for ad_id, year, geo, sex in clicks:
            if ad_id not in ad_ids:
                ad_ids.append(ad_id)
            i_index = ad_ids.index(ad_id)

            geo_vec = np.zeros(n_geo)
            geo_vec[int(geo) - 1] = 1.

            # normalized age in [0, 1]
            # clickgenerator.jl generates a birth year in [1930, 2000]
            age = 1. - ((2000 - year) / 70.)

            user = np.concatenate((np.array([age]), np.array([sex]), geo_vec))

            sample = {
                'y': 1,
                'u_index': u_index,
                'i_index': i_index,
                'user': user,
                'item': np.array([0.]),  # no detail about items
                'others': np.array([0.])
            }

            self.samples.append(sample)
            u_index += 1

        self.n_user = u_index
        self.n_item = 5  # 5 ad variants
        self.n_sample = len(self.samples)
        self.n_batch_train = int(self.n_sample * 0.2)  # 20% for pre-training to avoid cold-start
        self.n_batch_test = int(self.n_sample * 0.1)  # 10% for evaluation of pre-training
        self.n_test = self.n_sample - (self.n_batch_train + self.n_batch_test)
