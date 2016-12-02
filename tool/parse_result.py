import click
import numpy as np


def parse_result(filepath, window_size, at, n_item):
    f = open(filepath)
    lines = [[float(v) for v in l.rstrip().split('\t')] for l in f.readlines()]
    f.close()

    mat = np.array(lines)

    n_test = mat.shape[0]

    recalls = np.zeros(n_test)
    window = np.zeros(window_size)
    sum_window = 0.

    percentiles = np.zeros(n_test)

    ranks = mat[:, 1]
    for i, rank in enumerate(ranks):
        # increment a hit counter if i_index is in the top-{at} recommendation list
        # i.e. score the recommendation list based on the true observed item
        wi = i % window_size
        old = window[wi]
        new = 1. if (rank < at) else 0.
        window[wi] = new
        sum_window = sum_window - old + new
        recalls[i] = sum_window / min(i + 1, window_size)

        percentiles[i] = rank / (n_item - 1) * 100

    return {'top1_scores': mat[:, 0],
            'avg_recommend': np.mean(mat[:, 2]),
            'avg_update': np.mean(mat[:, 3]),
            'MPR': np.mean(percentiles),
            'incremental_recalls': recalls,
            'recall': np.mean((ranks < at).astype(np.int))}


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
