# coding: utf-8

import numpy as np
import pandas as pd


class LastFMConverter:

    """See also LastFM.ipynb for preprocessing.
    """

    def __init__(self):
        # contexts in this dataset
        # others: 1 normalized time
        # user: 1 standardized age, 1 sex, 16 for country
        # item: none
        self.contexts = {'others': 1, 'user': 18, 'item': 1}

        self.can_repeat = True

    def convert(self):
        """Create a list of samples and count number of users/items.

        """
        df_lastfm = pd.read_csv('converter/lastfm.tsv', delimiter='\t')

        self.samples = []
        self.dts = []

        countries = list(set(df_lastfm['country']))
        n_country = len(countries)  # 16 in total
        d_country = dict(zip(countries, range(n_country)))

        for i, row in df_lastfm.iterrows():
            country_vec = np.zeros(n_country)
            country_vec[d_country[row['country']]] = 1.

            user = np.concatenate((np.array([row['age']]), np.array([row['gender']]), country_vec))

            sample = {
                'y': 1,
                'u_index': row['u_index'],
                'i_index': row['i_index'],
                'user': user,
                'item': np.array([0.]),  # no detail about items
                'others': np.array([row['time']])
            }

            self.samples.append(sample)
            self.dts.append(row['dt'])

        self.n_user = len(set(df_lastfm['userid']))
        self.n_item = len(set(df_lastfm['track-id']))
        self.n_sample = len(self.samples)
        self.n_batch_train = int(self.n_sample * 0.2)  # 20% for pre-training to avoid cold-start
        self.n_batch_test = int(self.n_sample * 0.1)  # 10% for evaluation of pre-training
        self.n_test = self.n_sample - (self.n_batch_train + self.n_batch_test)
