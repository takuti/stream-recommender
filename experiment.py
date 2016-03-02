# coding: utf-8

import numpy as np
import time
from calendar import monthrange
from datetime import datetime, timedelta

from core.incremental_MF import IncrementalMF
from core.incremental_biasedMF import IncrementalBiasedMF
from core.incremental_FMs import IncrementalFMs

PATH_TO_RATINGS = '../../data/ml-1m/ratings.dat'

class Runner:
    def __init__(self):
        self.__prepare()

    def iMF(self, batch_flg=False):
        """Incremental Matrix Factorization

        Args:
            batch_flg (bool): choose whether a model is incrementally updated.
                True -- baseline
                False -- incremental matrix factorization

        Returns:
            list of float values: Simple Moving Averages.
            float: average time to recommend/update for one sample.

        """
        model = IncrementalMF(self.n_user, self.n_item)

        # pre-train
        model.fit(self.samples[:self.n_train])

        return model.evaluate(self.samples[self.n_train:], batch_flg)

    def biased_iMF(self):
        """Biased Incremental Matrix Factorizaton

        Returns:
           list of float values: Simple Moving Averages
           float: average time to recommend/update for one sample

        """
        model = IncrementalBiasedMF(self.n_user, self.n_item)
        model.fit(self.samples[:self.n_train])
        return model.evaluate(self.samples[self.n_train:])

    def iFMs(self, contexts=[]):
        """Incremental Factorization Machines

        Args:
            contexts (list): give contexts what a model incorporates as features.
                'dt' -- a model will be time-awared.

        Returns:
           list of float values: Simple Moving Averages
           float: average time to recommend/update for one sample

        """
        model = IncrementalFMs(self.n_user, self.n_item, contexts)
        model.fit(self.samples[:self.n_train])
        return model.evaluate(self.samples[self.n_train:])

    def __prepare(self):
        """Create a list of samples and count number of users/items.

        """
        self.__load_ratings()

        user_ids = []
        item_ids = []

        self.samples = []

        base_date = datetime(*time.localtime(self.ratings[0,3])[:6])
        self.dts = []

        for user_id, item_id, rating, timestamp in self.ratings:
            # give an unique user index
            if user_id not in user_ids: user_ids.append(user_id)
            u_index = user_ids.index(user_id)

            # give an unique item index
            if item_id not in item_ids: item_ids.append(item_id)
            i_index = item_ids.index(item_id)

            # delta days
            date = datetime(*time.localtime(timestamp)[:6])
            dt = self.__delta(base_date, date)

            sample = { 'u_index': u_index, 'i_index': i_index, 'dt': dt }

            self.samples.append(sample)
            self.dts.append(dt)

        self.n_user = len(user_ids)
        self.n_item = len(item_ids)
        self.n_sample = len(self.samples)
        self.n_train = int(self.n_sample * 0.2) # 20% for pre-training to avoid cold-start

    def __load_ratings(self):
        """Load all "positive" samples in the MovieLens 1M dataset.

        """
        ratings = []
        with open(PATH_TO_RATINGS) as f:
            lines = map(lambda l: map(int, l.rstrip().split('::')), f.readlines())
            for l in lines:
                # Since we consider positive-only feedback setting, ratings < 5 will be excluded.
                if l[2] == 5: ratings.append(l)
        self.ratings = np.asarray(ratings)

        # sorted by timestamp
        self.ratings = self.ratings[np.argsort(self.ratings[:,3])]

    def __delta(self, d1, d2, opt='d'):
        """Compute difference between given 2 dates in month/day.

        """
        delta = 0

        if opt == 'm':
            while True:
                mdays = monthrange(d1.year, d1.month)[1]
                d1 += timedelta(days=mdays)
                if d1 <= d2:
                    delta += 1
                else:
                    break
        else:
            delta = (d2 - d1).days

        return delta


def save(path, avgs, time):
    with open(path, 'w') as f:
        f.write('\n'.join(map(str, [time] + avgs)))

import click

models = ['baseline', 'iMF', 'biased-iMF', 'iFMs', 'iFMs-time-aware', 'all_MF']

@click.command()
@click.option('--model', type=click.Choice(models), default=models[0], help='Choose a factorization model')
def cli(model):
    exp = Runner()

    if model == 'all_MF':
        avgs, time = exp.iMF(batch_flg=True)
        save('results/baseline.txt', avgs, time)

        avgs, time = exp.iMF()
        save('results/iMF.txt', avgs, time)

        avgs, time = exp.biased_iMF()
        save('results/biased-iMF.txt', avgs, time)
    else:
        if model == 'baseline' or model == 'iMF':
            avgs, time = exp.iMF(batch_flg=True) if model == 'baseline' else exp.iMF()
        elif model == 'biased-iMF':
            avgs, time = exp.biased_iMF()
        else:
            avgs, time = exp.iFMs(['dt']) if model == 'iFMs-time-aware' else exp.iFMs()

        save('results/%s.txt' % model, avgs, time)

if __name__ == '__main__':
    cli()
