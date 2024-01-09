import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def partition_bins(boundary_scores, num_bins=5):
    sorted_indices = np.argsort(boundary_scores)
    bin_size = int(len(boundary_scores) / num_bins)
    return [sorted_indices[i * bin_size: (i + 1) * bin_size] for i in range(num_bins)]

def boundary_score_vs_asr(result_filename):
    print('plot ', result_filename)
    df = pd.read_csv(os.path.join('./result', result_filename))
    clean_pred = df['clean prediction'].values
    adv_pred = df['adv prediction'].values
    boundary_scores = df['boundary score'].values

    bins = partition_bins(boundary_scores, num_bins=5)

    x = [r'$\geq$' +f'{boundary_scores[b[0]]:.2f}\n' + r'$\leq$'+f'{boundary_scores[b[-1]]:.2f}' for b in bins]
    y = [np.sum(clean_pred[b] != adv_pred[b]) / len(clean_pred[b]) for b in bins]
    sns.set_theme()
    fig = plt.figure()
    ax = sns.barplot(x=x, y=y)
    ax.set_xlabel('Boundary score')
    ax.set_ylabel('ASR')
    fig.subplots_adjust(bottom=0.15)
    plt.show()


if __name__ == '__main__':
    boundary_score_vs_asr(sys.argv[1])
