import click
import numpy as np
from collections import deque


def measure(n_item, at, metric, rank):
    if metric == 'recall':
        return 1. if (rank < at) else 0.
    elif metric == 'precision':
        return 1 / at if (rank < at) else 0.
    elif metric == 'mpr':
        return rank / (n_item - 1) * 100


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
