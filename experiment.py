# coding: utf-8

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

import numpy as np
import time
from calendar import monthrange
from datetime import datetime, timedelta

from core.incremental_MF import IncrementalMF
from core.incremental_biasedMF import IncrementalBiasedMF
from core.incremental_FMs import IncrementalFMs

PATH_TO_RATINGS = '../../data/ml-1m/ratings.dat'
PATH_TO_MOVIES = '../../data/ml-1m/movies.dat'
PATH_TO_USERS = '../../data/ml-1m/users.dat'


class Runner:

    def __init__(self, method='SMA', limit=200000, n_trial=1, n_epoch=1):
        self.method = method

        # number of test samples
        self.limit = limit

        # number of trials for the incrementalRecall-based evaluation
        self.n_trial = n_trial

        # number of epochs for the batch training
        self.n_epoch = n_epoch

        self.__prepare()

    def iMF(self, is_static=False):
        """Incremental Matrix Factorization

        Args:
            is_static (bool): choose whether a model is incrementally updated.
                True -- baseline
                False -- incremental matrix factorization

        Returns:
            list of float values: Simple Moving Averages or avg. incrementalRecall.
            float: average time to recommend/update for one sample

        """
        logger.debug('%s-based evaluation of iMF' % self.method)

        def create():
            return IncrementalMF(self.n_user, self.n_item, is_static)

        model, res = self.__run(create)
        return res

    def biased_iMF(self):
        """Biased Incremental Matrix Factorizaton

        Returns:
            list of float values: Simple Moving Averages or avg. incrementalRecall.
            float: average time to recommend/update for one sample

        """
        logger.debug('%s-based evaluation of biased-iMF' % self.method)

        def create():
            return IncrementalBiasedMF(self.n_user, self.n_item)

        model, res = self.__run(create)

        logger.debug(
            'Regularization parameters: w0 = %s, w = %s, V = %s' % (
                model.l2_reg_w0,
                model.l2_reg_w,
                model.l2_reg_V))

        return res

    def iFMs(self, contexts=()):
        """Incremental Factorization Machines

        Args:
            contexts (tuple): give contexts what a model incorporates as features.
                'dt' -- a model will be time-awared.
                'genre' -- vector of a categorical variable for genres; 1-of-k numpy array (18,)
                'demographics' -- vector of user demographics; 1 for M/F, 1 for age group, 21 for a categorical variable for occupation

        Returns:
            list of float values: Simple Moving Averages or avg. incrementalRecall.
            float: average time to recommend/update for one sample

        """
        logger.debug('%s-based evaluation of iFMs' % self.method)

        def create():
            return IncrementalFMs(self.n_user, self.n_item, contexts)

        model, res = self.__run(create)

        logger.debug(
            'Regularization parameters: w0 = %s, w = %s, V = %s' % (
                model.l2_reg_w0,
                model.l2_reg_w,
                model.l2_reg_V))

        return res

    def __run(self, callback):
        """Test runner.

        Args:
            callback (function): Create a model used by this test run.

        Returns:
            instance of incremental model class: Created by the callback function.
            list of float values: Simple Moving Averages or avg. incrementalRecall.
            float: average time to recommend/update for one sample

        """
        batch_tail = self.n_batch_train + self.n_batch_test

        if self.method == 'SMA':
            model = callback()

            # pre-train
            model.fit(
                self.samples[:self.n_batch_train],
                self.samples[self.n_batch_train:batch_tail],
                n_epoch=self.n_epoch
            )

            res = model.evaluate_SMA(self.samples[batch_tail:batch_tail + self.n_test])
        elif self.method == 'recall':
            recalls = np.array([])
            s_time = 0.
            for i in range(self.n_trial):
                model = callback()
                model.fit(
                    self.samples[:self.n_batch_train],
                    self.samples[self.n_batch_train:batch_tail],
                    n_epoch=self.n_epoch
                )

                recall, avg_time = model.evaluate_recall(self.samples[batch_tail:batch_tail + self.n_test])
                logger.debug('Trial %d: recall = %.5f' % (i + 1, recall))

                recalls = np.append(recalls, recall)
                s_time += avg_time

            res = recalls, s_time / self.n_trial

        return model, res

    def __prepare(self):
        """Create a list of samples and count number of users/items.

        """
        self.__load_ratings()

        users = self.__load_users()
        movies = self.__load_movies()

        user_ids = []
        item_ids = []

        self.samples = []

        prev_date = datetime(*time.localtime(self.ratings[0, 3])[:6])

        head_date = datetime(*time.localtime(self.ratings[0, 3])[:6])
        self.dts = []

        for user_id, item_id, rating, timestamp in self.ratings:
            # give an unique user index
            if user_id not in user_ids:
                user_ids.append(user_id)
            u_index = user_ids.index(user_id)

            # give an unique item index
            if item_id not in item_ids:
                item_ids.append(item_id)
            i_index = item_ids.index(item_id)

            # delta days
            date = datetime(*time.localtime(timestamp)[:6])
            dt = self.__delta(prev_date, date)
            prev_date = date

            sample = {'u_index': u_index, 'i_index': i_index, 'dt': np.array([dt]), 'genre': movies[item_id], 'demographics': users[user_id]}

            self.samples.append(sample)
            self.dts.append(self.__delta(head_date, date))

        self.n_user = len(user_ids)
        self.n_item = len(item_ids)
        self.n_sample = len(self.samples)
        self.n_batch_train = int(self.n_sample * 0.3)  # 30% for pre-training to avoid cold-start
        self.n_batch_test = int(self.n_sample * 0.2)  # 20% for evaluation of pre-training
        self.n_test = min(self.n_sample - (self.n_batch_train + self.n_batch_test), self.limit)

        logger.debug('[exp] n_sample = %d; %d (30%%) + %d (20%%) + %d (50%%)' % (self.n_sample, self.n_batch_train, self.n_batch_test, self.n_test))
        logger.debug('[exp] n_user = %d, n_item = %d' % (self.n_user, self.n_item))

    def __load_movies(self):
        """Load movie genres as a context.

        Returns:
            dict of movie vectors: item_id -> numpy array (n_genre,)

        """
        with open(PATH_TO_MOVIES) as f:
            lines = map(lambda l: l.rstrip().split('::'), f.readlines())

        all_genres = ['Action',
                      'Adventure',
                      'Animation',
                      "Children's",
                      'Comedy',
                      'Crime',
                      'Documentary',
                      'Drama',
                      'Fantasy',
                      'Film-Noir',
                      'Horror',
                      'Musical',
                      'Mystery',
                      'Romance',
                      'Sci-Fi',
                      'Thriller',
                      'War',
                      'Western']
        n_genre = len(all_genres)

        movies = {}
        for item_id_str, title, genres in lines:
            movie_vec = np.zeros(n_genre)
            for genre in genres.split('|'):
                i = all_genres.index(genre)
                movie_vec[i] = 1.
            movies[int(item_id_str)] = movie_vec

        return movies

    def __load_users(self):
        """Load user demographics as contexts.User ID -> {sex (M/F), age (7 groupd), occupation(0-20; 21)}

        Returns:
            dict of user vectors: user_id -> numpy array (1+1+21,); (sex_flg + age_group + n_occupation, )

        """
        with open(PATH_TO_USERS) as f:
            lines = map(lambda l: l.rstrip().split('::'), f.readlines())

        ages = [1, 18, 25, 35, 45, 50, 56]

        users = {}
        for user_id_str, sex_str, age_str, occupation_str, zip_code in lines:
                user_vec = np.zeros(1 + 1 + 21)  # 1 categorical, 1 value, 21 categorical
                user_vec[0] = 0 if sex_str == 'M' else 1  # sex
                user_vec[1] = ages.index(int(age_str))  # age group (1, 18, ...)
                user_vec[2 + int(occupation_str)] = 1  # occupation (1-of-21)
                users[int(user_id_str)] = user_vec

        return users

    def __load_ratings(self):
        """Load all "positive" samples in the MovieLens 1M dataset.

        """
        ratings = []
        with open(PATH_TO_RATINGS) as f:
            lines = map(lambda l: map(int, l.rstrip().split('::')), f.readlines())
            for l in lines:
                # Since we consider positive-only feedback setting, ratings < 5 will be excluded.
                if l[2] == 5:
                    ratings.append(l)
        self.ratings = np.asarray(ratings)

        # sorted by timestamp
        self.ratings = self.ratings[np.argsort(self.ratings[:, 3])]

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
        f.write('\n'.join(map(str, np.append(time, avgs))))

import click

methods = ['SMA', 'recall']
models = ['baseline', 'iMF', 'biased-iMF', 'iFMs', 'all_MF', 'all_FMs']
contexts = ['dt', 'genre', 'demographics']


@click.command()
@click.option('--method', type=click.Choice(methods), default=methods[0], help='Choose an evaluation methodology.')
@click.option('--model', type=click.Choice(models), default=models[0], help='Choose a factorization model')
@click.option('--context', '-c', type=click.Choice(contexts), multiple=True, help='Choose contexts used by iFMs')
@click.option('--limit', default=200000, help='Number of test samples for evaluation')
@click.option('--n_trial', '-n', default=1, help='Number of trials for incrementallRecall-based evaluation.')
@click.option('--n_epoch', default=1, help='Number of epochs for batch training.')
def cli(method, model, context, limit, n_trial, n_epoch):
    exp = Runner(method=method, limit=limit, n_trial=n_trial, n_epoch=n_epoch)

    if model == 'all_MF':
        avgs, time = exp.iMF(is_static=True)
        save('results/baseline_' + method + '.txt', avgs, time)

        avgs, time = exp.iMF()
        save('results/iMF_' + method + '.txt', avgs, time)

        avgs, time = exp.biased_iMF()
        save('results/biased-iMF_' + method + '.txt', avgs, time)
    elif model == 'all_FMs':
        avgs, time = exp.iFMs(())
        save('results/iFMs_no_context_' + method + '.txt', avgs, time)

        avgs, time = exp.iFMs(('dt', 'genre', 'demographics'))
        save('results/iFMs_contexts_' + method + '.txt', avgs, time)
    else:
        if model == 'baseline' or model == 'iMF':
            avgs, time = exp.iMF(is_static=True) if model == 'baseline' else exp.iMF()
        elif model == 'biased-iMF':
            avgs, time = exp.biased_iMF()
        elif model == 'iFMs':
            model = model + '_' + '-'.join(context)  # update output filename depending on contexts
            avgs, time = exp.iFMs(context)

        save('results/%s_%s.txt' % (model, method), avgs, time)

if __name__ == '__main__':
    cli()
