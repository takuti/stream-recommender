import click
import numpy as np


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


def parse_result(filepath, window_size, n_item, at=10):
    f = open(filepath)
    lines = [[float(v) for v in l.rstrip().split('\t')] for l in f.readlines()]
    f.close()

    mat = np.array(lines)
    n_test = mat.shape[0]
    ranks = mat[:, 1]

    windows = {'recall': np.zeros(window_size),
               'precision': np.zeros(window_size),
               'map': np.zeros(window_size),
               'mrr': np.zeros(window_size),
               'auc': np.zeros(window_size),
               'mpr': np.zeros(window_size),
               'ndcg': np.zeros(window_size)}

    sums = {'recall': 0,
            'precision': 0,
            'map': 0,
            'mrr': 0,
            'auc': 0,
            'mpr': 0,
            'ndcg': 0}

    res = {'recall': np.zeros(n_test),
           'precision': np.zeros(n_test),
           'map': np.zeros(n_test),
           'mrr': np.zeros(n_test),
           'auc': np.zeros(n_test),
           'mpr': np.zeros(n_test),
           'ndcg': np.zeros(n_test)}

    for i, rank in enumerate(ranks):
        for metric in ['recall', 'precision', 'map', 'mrr', 'auc', 'mpr', 'ndcg']:
            wi = i % window_size

            old = windows[metric][wi]

            new = measure(n_item, at, metric, rank)
            windows[metric][wi] = new

            sums[metric] = sums[metric] - old + new

            res[metric][i] = sums[metric] / min(i + 1, window_size)

    return {'top1_scores': mat[:, 0],
            'avg_recommend': np.mean(mat[:, 2]),
            'avg_update': np.mean(mat[:, 3]),
            'recall': res['recall'],
            'precision': res['precision'],
            'map': res['map'],
            'mrr': res['mrr'],
            'auc': res['auc'],
            'mpr': res['mpr'],
            'ndcg': res['ndcg']}


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
