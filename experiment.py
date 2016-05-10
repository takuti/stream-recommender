# coding: utf-8

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

import numpy as np
from functools import partial

from core.incremental_MF import IncrementalMF
from core.incremental_FMs import IncrementalFMs
from core.online_sketch import OnlineSketch
from core.random import Random

from converter.converter import Converter


class Runner:

    def __init__(self, dataset='ML1M', window_size=5000, n_epoch=1):
        self.window_size = window_size

        # number of epochs for the batch training
        self.n_epoch = n_epoch

        # load dataset
        self.data = Converter().convert(dataset=dataset)

        logger.debug('[exp] %s | window_size = %d, n_epoch = %d' % (dataset, window_size, n_epoch))
        logger.debug('[exp] n_sample = %d; %d (30%%) + %d (20%%) + %d (50%%)' % (
            self.data.n_sample, self.data.n_batch_train, self.data.n_batch_test, self.data.n_test))
        logger.debug('[exp] n_user = %d, n_item = %d' % (self.data.n_user, self.data.n_item))

    def iMF(self, is_static=False):
        """Incremental Matrix Factorization

        Args:
            is_static (bool): choose whether a model is incrementally updated.
                True -- baseline
                False -- incremental matrix factorization

        Returns:
            list of float values: Simple Moving Averages (i.e. incremental recall).
            float: average time to recommend/update for one sample

        """
        if is_static:
            logger.debug('# static MF')
        else:
            logger.debug('# iMF')

        def create():
            return IncrementalMF(self.data.n_item, is_static)

        model, res = self.__run(create)
        return res

    def iFMs(self, is_context_aware=False):
        """Incremental Factorization Machines

        Args:
            is_context_aware (bool): Choose whether a feature vector incorporates contextual variables of the dataset.

        Returns:
            list of float values: Simple Moving Averages (i.e. incremental recall).
            float: average time to recommend/update for one sample

        """
        logger.debug('# iFMs')

        def create():
            partial_iFMs = partial(
                IncrementalFMs,
                n_item=self.data.n_item)

            if is_context_aware:
                return partial_iFMs(contexts=self.data.contexts)
            else:
                return partial_iFMs(contexts=[])

        model, res = self.__run(create)

        logger.debug(
            'Regularization parameters: w0 = %s, w = %s, V = %s' % (
                model.l2_reg_w0,
                model.l2_reg_w,
                model.l2_reg_V))

        return res

    def sketch(self):
        """Online Matrix Sketching

        Returns:
            list of float values: Simple Moving Averages (i.e. incremental recall).
            float: average time to recommend/update for one sample

        """
        logger.debug('# matrix sketching')

        def create():
            return OnlineSketch(self.data.n_item)

        model, res = self.__run(create)

        return res

    def random(self):
        """Random baseline

        Returns:
            list of float values: Simple Moving Averages (i.e. incremental recall).
            float: average time to recommend/update for one sample

        """
        logger.debug('# random baseline')

        def create():
            return Random(self.data.n_item)

        model, res = self.__run(create)

        return res

    def __run(self, callback):
        """Test runner.

        Args:
            callback (function): Create a model used by this test run.

        Returns:
            instance of incremental model class: Created by the callback function.
            list of float values: Simple Moving Averages (i.e. incremental recall).
            float: average time to recommend/update for one sample

        """
        batch_tail = self.data.n_batch_train + self.data.n_batch_test

        model = callback()

        # pre-train
        # 30% for batch training | 20% for batch evaluate
        model.fit(
            self.data.samples[:self.data.n_batch_train],
            self.data.samples[self.data.n_batch_train:batch_tail],
            n_epoch=self.n_epoch
        )

        # 70% incremental evaluation and updating (20% batch evaluation samples + 50% remainders)
        res = model.evaluate(self.data.samples[self.data.n_batch_train:], window_size=self.window_size)

        return model, res


def save(path, avgs, time):
    with open(path, 'w') as f:
        f.write('\n'.join(map(str, np.append(time, avgs))))

import click

models = ['static-MF', 'iMF', 'iFMs', 'sketch', 'random']
datasets = ['ML1M', 'ML100k', 'LastFM']


@click.command()
@click.option('--model', type=click.Choice(models), default=models[0], help='Choose a factorization model')
@click.option('--dataset', type=click.Choice(datasets), default=datasets[0], help='Choose a dataset')
@click.option('--context/--no-context', default=False, help='Choose whether a feature vector for iFMs incorporates contextual variables.')
@click.option('--window_size', default=5000, help='Window size of the simple moving average for incremental evaluation.')
@click.option('--n_epoch', default=1, help='Number of epochs for batch training.')
def cli(model, dataset, context, window_size, n_epoch):
    exp = Runner(dataset=dataset, window_size=window_size, n_epoch=n_epoch)

    if model == 'static-MF' or model == 'iMF':
        avgs, time = exp.iMF(is_static=True) if model == 'static-MF' else exp.iMF()
    elif model == 'sketch':
        avgs, time = exp.sketch()
    elif model == 'random':
        avgs, time = exp.random()
    elif model == 'iFMs':
        model += ('_context_aware' if context else '_no_context')  # update output filename depending on contexts
        avgs, time = exp.iFMs(is_context_aware=context)

    save('results/%s_%s_%s.txt' % (dataset, model, window_size), avgs, time)

if __name__ == '__main__':
    cli()
