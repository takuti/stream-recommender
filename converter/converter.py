# coding: utf-8

from MovieLens1M import MovieLens1MConverter
from MovieLens100k import MovieLens100kConverter


class Converter:

    def __init__(self):
        pass

    def convert(self, dataset='ML1M'):
        """Convert a specified dataset to be used by experiments.

        Args:
            dataset (str): Dataset name.

        Returns:
            instance of a converter: Its instance variables are used by experiments.

        """
        if dataset == 'ML1M':
            c = MovieLens1MConverter()
            c.convert(is_positive_only=False)
        elif dataset == 'ML1M+':
            c = MovieLens1MConverter()
            c.convert(is_positive_only=True)
        elif dataset == 'ML100k':
            c = MovieLens100kConverter()
            c.convert(is_positive_only=False)
        elif dataset == 'ML100k+':
            c = MovieLens100kConverter()
            c.convert(is_positive_only=True)

        return c
