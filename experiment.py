# coding: utf-8

import click

from logging import getLogger, StreamHandler, Formatter, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setFormatter(Formatter('[%(process)d] %(message)s'))
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

from core.incremental_MF import IncrementalMF
from core.incremental_FMs import IncrementalFMs
from core.online_sketch import OnlineSketch
from core.random import Random
from core.popular import Popular

from converter.converter import Converter


class Runner:

    def __init__(self, params, dataset='ML1M', n_epoch=1):
        self.params = params

        # number of epochs for the batch training
        self.n_epoch = n_epoch

        # load dataset
        self.data = Converter().convert(dataset=dataset)

        logger.debug('[exp] %s | n_epoch = %d' % (dataset, n_epoch))
        logger.debug('[exp] n_sample = %d; %d (20%%) + %d (10%%) + %d (70%%)' % (
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
            return IncrementalMF(is_static,
                                 int(self.params['k']),
                                 self.params['l2_reg'],
                                 self.params['learn_rate'])

        model, res = self.__run(create)
        return res

    def iFMs(self, is_context_aware=False, is_static=False):
        """Incremental Factorization Machines

        Args:
            is_context_aware (bool): Choose whether a feature vector incorporates contextual variables of the dataset.

        Returns:
            list of float values: Simple Moving Averages (i.e. incremental recall).
            float: average time to recommend/update for one sample

        """
        if is_static:
            logger.debug('# static FMs')
        else:
            logger.debug('# iFMs')

        def create():
            return IncrementalFMs(self.data.contexts, is_static,
                                  int(self.params['k']),
                                  self.params['l2_reg_w0'],
                                  self.params['l2_reg_w'],
                                  self.params['l2_reg_v'],
                                  self.params['learn_rate'])

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
            return OnlineSketch(contexts=self.data.contexts,
                                k=int(self.params['k']),
                                ell=int(self.params['ell']))

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
            return Random()

        model, res = self.__run(create)

        return res

    def popular(self):
        """Popularity (non-personalized) baseline

        Returns:
            list of float values: Simple Moving Averages (i.e. incremental recall).
            float: average time to recommend/update for one sample

        """
        logger.debug('# popularity baseline')

        def create():
            return Popular()

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
        model.set_can_repeat(self.data.can_repeat)

        # pre-train
        # 20% for batch training | 10% for batch evaluate
        # after the batch training, 10% samples are used for incremental updating
        model.fit(
            self.data.samples[:self.data.n_batch_train],
            self.data.samples[self.data.n_batch_train:batch_tail],
            n_epoch=self.n_epoch
        )

        # 70% incremental evaluation and updating
        res = model.evaluate(self.data.samples[batch_tail:])

        return model, res


@click.command()
@click.option('--config', '-f', help='Give a path to your config file.')
def cli(config):

    def save(path, res_tuples):
        f = open(path, 'w')
        lines = ['\t'.join([str(v) for v in t]) for t in res_tuples]
        f.write('\n'.join(lines))
        f.close()

    # parse given config file
    parser = configparser.ConfigParser()
    parser.read(config)

    c = parser['Common']
    dataset = c.get('Dataset')  # ['ML1M', 'ML100k', 'LastFM', 'click']
    n_trial = c.getint('Trial', 1)

    # ['static-MF', 'iMF', 'static-FMs', 'iFMs', 'sketch', 'random', 'popular']
    m = parser['Model']
    model = m['Name']
    n_epoch = m.getint('Epoch', 1)

    if 'Parameters' in parser:
        params = dict([(k, float(v)) for k, v in parser['Parameters'].items()])
    else:
        params = {}

    exp = Runner(params=params, dataset=dataset, n_epoch=n_epoch)

    for i in range(n_trial):
        if model == 'static-MF' or model == 'iMF':
            res = exp.iMF(is_static=True) if model == 'static-MF' else exp.iMF()
        elif model == 'sketch':
            res = exp.sketch()
        elif model == 'random':
            res = exp.random()
        elif model == 'popular':
            res = exp.popular()
        elif model == 'static-FMs' or model == 'iFMs':
            res = exp.iFMs(is_static=True) if model == 'static-FMs' else exp.iFMs()

        save('results/%s_%s_%s.txt' % (dataset, model, i + 1), list(res))


if __name__ == '__main__':
    import configparser
    cli()
