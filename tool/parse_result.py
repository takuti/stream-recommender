import click
import numpy as np
from collections import deque


def measure(n_item, at, metric, rank):
    # rank is in [0, n_item)
    if metric == 'recall':
        return 1. if (rank < at) else 0.
    elif metric == 'precision':
        return 1 / at if (rank < at) else 0.
    elif metric == 'map' or metric == 'mrr':
        return 1 / (rank + 1)
    elif metric == 'auc':
        correct = n_item - (rank + 1)
        pairs = 1 * (n_item - 1)
        return correct / pairs
    elif metric == 'mpr':
        return rank / (n_item - 1) * 100
    elif metric == 'ndcg':
        dcg = 1 / np.log2(rank + 2) if (rank < at) else 0.
        idcg = sum([1 / np.log2(n + 1) for n in range(1, at + 1)])
        return dcg / idcg


def parse_result(filepath, window_size, n_item, at=10, metric='recall'):
    f = open(filepath)
    lines = [[float(v) for v in l.rstrip().split('\t')] for l in f.readlines()]
    f.close()

    mat = np.array(lines)
    ranks = mat[:, 1]

    window = deque(maxlen=window_size)
    res = np.array([])

    for i, rank in enumerate(ranks):
        v = measure(n_item, at, metric, rank)
        window.append(v)
        res = np.append(res, np.mean(window))

    return {'top1_scores': mat[:, 0],
            'avg_recommend': np.mean(mat[:, 2]),
            'avg_update': np.mean(mat[:, 3]),
            'res': res}


@click.command()
@click.option('--filepath', '-f', help='Give a path to your result file.')
@click.option('--window_size', '-w', default=1, help='Size of a window for stream evaluation.')
@click.option('--at', default=10, help='Evaluate recall@{at}.')
@click.option('--n_item', default=5, help='Number of items on the dataset.')
def cli(filepath, window_size, at, n_item):
    res = parse_result(filepath, window_size, at, n_item)

    f = open(filepath + '.evaluate.txt', 'w')
    f.write(res['avg_recommend'] + '\n')
    f.write(res['avg_update'] + '\n')
    f.write(res['MPR'] + '\n')
    f.write('\n'.join(map(str, res['incremental_recalls'])))
    f.close()


if __name__ == '__main__':
    cli()
