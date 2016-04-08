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
from core.incremental_biasedMF import IncrementalBiasedMF
from core.incremental_FMs import IncrementalFMs

from converter.converter import Converter


class Runner:

    def __init__(self, method='SMA', dataset='ML1M', n_trial=1, n_epoch=1):
        self.method = method

        # number of trials for the incrementalRecall-based evaluation
        self.n_trial = n_trial

        # number of epochs for the batch training
        self.n_epoch = n_epoch

        # load dataset
        self.data = Converter().convert(dataset=dataset)

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
            list of float values: Simple Moving Averages or avg. incrementalRecall.
            float: average time to recommend/update for one sample

        """
        logger.debug('%s-based evaluation of iMF' % self.method)

        def create():
            return IncrementalMF(self.data.n_user, self.data.n_item, self.data.is_positive_only, is_static)

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
            return IncrementalBiasedMF(self.data.n_user, self.data.n_item, self.data.is_positive_only)

        model, res = self.__run(create)

        logger.debug(
            'Regularization parameters: w0 = %s, w = %s, V = %s' % (
                model.l2_reg_w0,
                model.l2_reg_w,
                model.l2_reg_V))

        return res

    def iFMs(self, is_context_aware=False):
        """Incremental Factorization Machines

        Args:
            is_context_aware (bool): Choose whether a feature vector incorporates contextual variables of the dataset.

        Returns:
            list of float values: Simple Moving Averages or avg. incrementalRecall.
            float: average time to recommend/update for one sample

        """
        logger.debug('%s-based evaluation of iFMs' % self.method)

        def create():
            partial_iFMs = partial(
                IncrementalFMs,
                n_user=self.data.n_user,
                n_item=self.data.n_item,
                is_positive_only=self.data.is_positive_only)

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

    def __run(self, callback):
        """Test runner.

        Args:
            callback (function): Create a model used by this test run.

        Returns:
            instance of incremental model class: Created by the callback function.
            list of float values: Simple Moving Averages or avg. incrementalRecall.
            float: average time to recommend/update for one sample

        """
        batch_tail = self.data.n_batch_train + self.data.n_batch_test

        if self.method == 'SMA':
            model = callback()

            # pre-train
            model.fit(
                self.data.samples[:self.data.n_batch_train],
                self.data.samples[self.data.n_batch_train:batch_tail],
                n_epoch=self.n_epoch
            )

            res = model.evaluate_SMA(self.data.samples[batch_tail:batch_tail + self.data.n_test])
        elif self.method == 'recall':
            recalls = np.array([])
            s_time = 0.
            for i in range(self.n_trial):
                model = callback()
                model.fit(
                    self.data.samples[:self.data.n_batch_train],
                    self.data.samples[self.data.n_batch_train:batch_tail],
                    n_epoch=self.n_epoch
                )

                recall, avg_time = model.evaluate_recall(self.data.samples[batch_tail:batch_tail + self.data.n_test])
                logger.debug('Trial %d: recall = %.5f' % (i + 1, recall))

                recalls = np.append(recalls, recall)
                s_time += avg_time

            res = recalls, s_time / self.n_trial

        return model, res


def save(path, avgs, time):
    with open(path, 'w') as f:
        f.write('\n'.join(map(str, np.append(time, avgs))))

import click

models = ['baseline', 'iMF', 'biased-iMF', 'iFMs', 'all_MF', 'all_FMs']
methods = ['SMA', 'recall']
datasets = [
    'ML1M',     # explicit rating feedback (w/ contexts)
    'ML100K',   # explicit rating feedback (w/ contexts)
    'ML1M+',    # positive-only feedback (w/ contexts)
    'ML100K+',  # positive-only feedback (w/ contexts)
    'LastFM',   # positive-only feedback (w/ contexts)
    'Epinions'  # explicit rating feedback (w/o contexts)
]


@click.command()
@click.option('--model', type=click.Choice(models), default=models[0], help='Choose a factorization model')
@click.option('--method', type=click.Choice(methods), default=methods[0], help='Choose an evaluation methodology.')
@click.option('--dataset', type=click.Choice(datasets), default=datasets[0], help='Choose a dataset')
@click.option('--context/--no-context', default=False, help='Choose whether a feature vector for iFMs incorporates contextual variables.')
@click.option('--n_trial', '-n', default=1, help='Number of trials for incrementallRecall-based evaluation.')
@click.option('--n_epoch', default=1, help='Number of epochs for batch training.')
def cli(model, method, dataset, context, n_trial, n_epoch):
    exp = Runner(method=method, dataset=dataset, n_trial=n_trial, n_epoch=n_epoch)

    if model == 'all_MF':
        avgs, time = exp.iMF(is_static=True)
        save('results/baseline_' + method + '.txt', avgs, time)

        avgs, time = exp.iMF()
        save('results/iMF_' + method + '.txt', avgs, time)

        avgs, time = exp.biased_iMF()
        save('results/biased-iMF_' + method + '.txt', avgs, time)
    elif model == 'all_FMs':
        avgs, time = exp.iFMs(is_context_aware=False)
        save('results/iFMs_no_context_' + method + '.txt', avgs, time)

        avgs, time = exp.iFMs(is_context_aware=True)
        save('results/iFMs_context_aware_' + method + '.txt', avgs, time)
    else:
        if model == 'baseline' or model == 'iMF':
            avgs, time = exp.iMF(is_static=True) if model == 'baseline' else exp.iMF()
        elif model == 'biased-iMF':
            avgs, time = exp.biased_iMF()
        elif model == 'iFMs':
            model += ('_context_aware' if context else '_no_context')  # update output filename depending on contexts
            avgs, time = exp.iFMs(is_context_aware=context)

        save('results/%s_%s.txt' % (model, method), avgs, time)

if __name__ == '__main__':
    cli()
