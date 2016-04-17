# coding: utf-8

"""Preliminarily, a subset of the LastFM dataset is extracted.

"""

import numpy as np
import pandas as pd


class LastFMConverter:

    def __init__(self):
        self.path = '../../data/lastfm-dataset-1K/lastfm-600k.tsv'

        # contexts in this dataset
        # 1 delta time, 1 age, 1 M/F, 23 for country
        self.contexts = [('dt', 1), ('age', 1), ('sex', 1), ('country', 23)]

    def convert(self):
        """Create a list of samples and count number of users/items.

        """
        df_lastfm_600k = pd.read_csv('../../data/lastfm-dataset-1K/lastfm-600k.tsv', delimiter='\t')

        self.samples = []
        self.dts = []

        countries = list(set(df_lastfm_600k['country']))
        d_country = dict(zip(countries, range(len(countries))))

        for i, row in df_lastfm_600k.iterrows():
            country_vec = np.zeros(23)
            country_vec[d_country[row['country']]] = 1.

            sex = 1. if row['gender'] == 'm' else 0.

            sample = {
                'y': 1,
                'u_index': row['u_index'],
                'i_index': row['i_index'],
                'dt': np.array([row['dt']]),
                'age': np.array([row['age']]),
                'sex': np.array([sex]),
                'country': country_vec
            }

            self.samples.append(sample)
            self.dts.append(row['dt'])

        self.n_user = len(set(df_lastfm_600k['userid']))
        self.n_item = len(set(df_lastfm_600k['track-id']))
        self.n_sample = len(self.samples)
        self.n_batch_train = int(self.n_sample * 0.3)  # 30% for pre-training to avoid cold-start
        self.n_batch_test = int(self.n_sample * 0.2)  # 20% for evaluation of pre-training
        self.n_test = self.n_sample - (self.n_batch_train + self.n_batch_test)
