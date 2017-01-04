# conding: utf-8

from flurs.data.entity import User, Item, Event
import numpy as np
import os


class SyntheticClickConverter:

    """Dataset will be generated as:

    $ julia tool/clickgenerator.jl > data/click.tsv
    """

    def __init__(self):
        # contexts in this dataset
        # others: none
        # user: 1 normalized age, 1 sex, 50 for living state
        # item: 3 garegories
        self.contexts = {'others': 1, 'user': 52, 'item': 3}

        # 3 ad categories (e.g. life, tech, money) for 5 ad
        # (Google also has ad interest categories: https://support.google.com/ads/answer/2842480?hl=en)
        self.categories = [0, 0, 2, 2, 1]

        self.can_repeat = True

    def convert(self):
        """Create a list of samples and count number of users/items.

        """

        clicks = []
        with open(os.path.join(os.path.dirname(__file__), '../data/click.tsv')) as f:
            clicks = list(map(lambda l: list(map(int, l.rstrip().split('\t'))), f.readlines()))

        self.samples = []

        u_index = 0  # each sample indicates different visitors
        n_geo = 50  # 50 states in US

        ad_ids = []
        ad_categories = []

        for ad_id, year, geo, sex in clicks:
            if ad_id not in ad_ids:
                ad_ids.append(ad_id)
                ad_categories.append(self.categories[ad_id])
            i_index = ad_ids.index(ad_id)

            geo_vec = np.zeros(n_geo)
            geo_vec[geo - 1] = 1.

            # normalized age in [0, 1]
            # clickgenerator.jl generates a birth year in [1930, 2000]
            age = 1. - ((2000 - year) / 70.)

            user = User(0, np.concatenate((np.array([age]), np.array([sex]), geo_vec)))

            # category vector
            category = np.zeros(3)
            category[ad_categories[i_index]] = 1

            item = Item(i_index, category)

            sample = Event(user, item, 1.)
            self.samples.append(sample)

            u_index += 1

        self.n_user = u_index
        self.n_item = 5  # 5 ad variants
        self.n_sample = len(self.samples)
        self.n_batch_train = int(self.n_sample * 0.2)  # 20% for pre-training to avoid cold-start
        self.n_batch_test = int(self.n_sample * 0.1)  # 10% for evaluation of pre-training
        self.n_test = self.n_sample - (self.n_batch_train + self.n_batch_test)
