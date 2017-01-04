# coding: utf-8

from flurs.data.entity import User, Item, Event
import numpy as np
import pandas as pd
import os


class LastFMConverter:

    """See `notebook/LastFM.ipynb` to know how `lastfm.tsv` was generated.
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
        df_lastfm = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/lastfm.tsv'), delimiter='\t')

        self.samples = []
        self.dts = []

        # number of artists will be dimension of item contexts
        n_artist = len(set(df_lastfm['artist_index']))
        self.contexts['item'] = n_artist

        countries = list(set(df_lastfm['country']))
        n_country = len(countries)  # 16 in total
        d_country = dict(zip(countries, range(n_country)))

        for i, row in df_lastfm.iterrows():
            country_vec = np.zeros(n_country)
            country_vec[d_country[row['country']]] = 1.

            user = User(row['u_index'],
                        np.concatenate((np.array([row['age']]), np.array([row['gender']]), country_vec)))

            artist_vec = np.zeros(n_artist)
            artist_vec[row['artist_index']] = 1

            item = Item(row['i_index'], artist_vec)

            sample = Event(user, item, 1., np.array([row['time']]))
            self.samples.append(sample)

            self.dts.append(row['dt'])

        self.n_user = len(set(df_lastfm['userid']))
        self.n_item = len(set(df_lastfm['track-id']))
        self.n_sample = len(self.samples)
        self.n_batch_train = int(self.n_sample * 0.2)  # 20% for pre-training to avoid cold-start
        self.n_batch_test = int(self.n_sample * 0.1)  # 10% for evaluation of pre-training
        self.n_test = self.n_sample - (self.n_batch_train + self.n_batch_test)
