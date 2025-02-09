import os
import math
import argparse
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

def incomplete_ratio_result(result_filename, dataset):
    df = pd.read_csv(result_filename)
    df['tpr'] = df['tp'] / (df['tp'] + df['fn'])
    df = df[df['method'] != 'ig']
    df = df[df['method'] != 'rnd']

    custom = {
        "axes.edgecolor": "gray", 
        # "grid.linestyle": "dashed", 
        "grid.color": "gray",
    } 
    sns.set_style("white", rc = custom)
    plt.figure()
    plt.rc('axes', labelsize=24)
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=24)

    # hue_order = ['ours', 'nettack', 'minmax']
    hue_order = ['ours', 'nettack', 'sga', 'fga', 'rnd']
    c = sns.color_palette('Set1', n_colors=5)
    ax = sns.lineplot(x='p', y='tpr', data=df, hue='method', hue_order=hue_order,
                      palette=c, errorbar=None, style='method', 
                      markers=True, dashes=False, markersize=10)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    labels = plt.gca().get_legend_handles_labels()
    print(labels)

    plt.xlabel('Incompleteness ratio')
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    # plt.xlabel('p')
    # plt.ylim(0.4, 1.0)
    plt.ylabel('True positive probability')
    plt.legend([], [], frameon=False)
    plt.savefig(f'./figures/incomplete_ratio_{dataset}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    ax_legend.axis('off')
    legend_names = ['PANDA (ours)', 'Nettack', 'SGAttack', 'FGA', 'RND']
    ax_legend.legend(labels[0], legend_names, loc='center', ncol=5)
    fig_legend.savefig('./figures/incomplete_ratio_legend.pdf', dpi=300, bbox_inches='tight')

def num_pert_edges_result(result_filename, dataset):
    df = pd.read_csv(result_filename)
    df = df[df['method'] == 'ours']
    df['num_edges'] = df['num_pert_edges'] * 2

    custom = {
        "axes.edgecolor": "gray", 
        # "grid.linestyle": "dashed", 
        "grid.color": "gray",
    } 
    sns.set_style("white", rc = custom)
    plt.rc('axes', labelsize=24)
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=24)

    # hue_order = ['ours', 'nettack', 'sga', 'fga']
    c = sns.color_palette('Set1', n_colors=4)
    fig, ax1 = plt.subplots()
    # # ax = sns.lineplot(x='num_pert_edges', y='tpr', data=df, hue='method', hue_order=hue_order,
    # #                   palette=c, errorbar=None, style='method', 
    # #                   markers=["s", "P", "X", "o"], dashes=False, markersize=10)
    # ax = sns.lineplot(x='num_pert_edges', y='tnr', data=df, hue='method', hue_order=hue_order,
    #                   palette=c, errorbar=None, style='method', 
    #                   markers=["o", "s", "P", "X"], dashes=False, markersize=10)
    sns.lineplot(x='num_edges', y='tpr', data=df, marker='o', markers=True, dashes=False,
                 markersize=10, color=c[0], label='TPR', ax=ax1, legend=False)
    ax1.set_ylabel(r'$P_{TP}$')
    ax1.set_xlabel('Number of challenge edges')
    ax1.tick_params(axis='y')
    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax1.legend([], [], frameon=False)

    ax2 = ax1.twinx()
    sns.lineplot(x='num_edges', y='tnr', data=df, marker='X', markers=True, dashes=False,
                    markersize=10, color=c[1], label='TNR', ax=ax2, legend=False)
    ax2.set_ylabel(r'$P_{TN}$')
    ax2.tick_params(axis='y')
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax2.legend([], [], frameon=False)

    # plt.xlabel('Number of challenge edges')
    plt.xticks([10, 20, 30, 40, 50])
    # plt.xlabel('p')
    # plt.ylim(0.4, 1.0)
    # plt.ylabel('True positive probability')
    # plt.ylabel('True negative probability')
    # plt.legend([], [], frameon=False)
    plt.savefig(f'./figures/num_pert_edges_tpr_tnr_{dataset}.pdf', dpi=300, bbox_inches='tight')
    # plt.show()

    # fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    # ax_legend.axis('off')
    # legend_names = ['PANDA (ours)', 'Nettack', 'SGAttack', 'FGA']
    # ax_legend.legend(labels[0], legend_names, loc='center', ncol=4)
    # fig_legend.savefig('./figures/num_pert_edges_legend.pdf', dpi=300, bbox_inches='tight')


def efficiency(result_filename, type):
    df = pd.read_csv(os.path.join('./result', result_filename))
    # df['preparation time'] = df['preparation time'] / 1000
    sns.set_style('white')
    plt.figure(figsize=(4, 6))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    c = sns.color_palette('Set1', n_colors=3)
    ax = sns.barplot(x='B', y='total_preparation_time', data=df, hue='method', errorbar=None,
                      palette=[c[1], c[2], c[0]], hue_order=['nettack', 'minmax', 'ours'])

    ax.legend_.set_title(None)
    # ax.legend_.set_labels(['IVEU(ours)', 'Nettack', 'MinMax'])
    ax.legend(['Nettack', 'MinMax', 'IVEU (ours)'])
    plt.xlabel('Verification budget B')
    plt.ylabel('Total preparation time (s)')
    plt.savefig(f'./figures/efficiency_{type}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def sensitivity(result_filename, type):
    df = pd.read_csv(result_filename)
    df['tpr'] = df['tp'] / (df['tp'] + df['fn'])
    df['tnr'] = df['tn'] / (df['tn'] + df['fp'])


    sns.set_style('white')
    plt.figure(figsize=(4, 5))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    # c = sns.color_palette('Set2', n_colors=3)
    if type == 'b':
        _df = df[df['b'] <= 25]
        _df = _df.melt(id_vars='b', value_vars=['tpr', 'tnr'], var_name='metric', value_name='y')
        ax = sns.lineplot(x='b', y='y', data=_df, hue='metric', 
                        errorbar=None, style='metric', 
                        markers=True, dashes=False, markersize=10)
        labels = plt.gca().get_legend_handles_labels()
        ax.set(xlabel=None)
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        # plt.xlabel('Verification budget B')
        plt.xticks([5, 10, 15, 20, 25])
    elif type == 'm':
        _df = df.melt(id_vars='m', value_vars=['tpr', 'tnr'], var_name='metric', value_name='y')
        ax = sns.lineplot(x='m', y='y', data=_df, hue='metric', 
                        errorbar=None, style='metric', 
                        markers=True, dashes=False, markersize=10)
        labels = plt.gca().get_legend_handles_labels()

        plt.xlabel('Search space M')
        plt.xticks([1, 2, 3, 4, 5])

    elif type == 't':
        _df = df.melt(id_vars='t', value_vars=['tpr', 'tnr'], var_name='metric', value_name='y')
        ax = sns.lineplot(x='t', y='y', data=_df, hue='metric', 
                        errorbar=None, style='metric', 
                        markers=True, dashes=False, markersize=10)
        ax.set(xlabel=None)
        labels = plt.gca().get_legend_handles_labels()
        plt.xticks([10, 20, 30, 40, 50])
        plt.ylim(0.90, 1.005)
        plt.yticks([0.85, 0.90, 0.95, 1.0])
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    
    # plt.xlabel('p')
    plt.ylabel('TPR/TNR')
    plt.legend([], [], frameon=False)
    plt.savefig(f'./figures/sensitivity_{type}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    ax_legend.axis('off')
    legend_names = ['TPR', 'TNR']
    ax_legend.legend(labels[0], legend_names, loc='center', ncol=3)
    fig_legend.savefig('./figures/sensitivity_legend.pdf', dpi=300, bbox_inches='tight')


def probabilistic_guarantee():
    N = 10 ** 3
    x = np.linspace(0, 100, 100)

    sns.set_style('white')
    plt.figure()
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    for k in [0.05, 0.10, 0.20, 0.50]:
        y = 1 - ((N - k * N) / N) ** x
        sns.lineplot(x=x, y=y, label=f'{k:.0%} as challenge edges', legend=True)
    plt.xlabel('Number of cheating unlearning requests (m)')
    plt.ylabel('Verification probability')
    # plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    # ax.set_xticklabels(['{:,.0%}'.format(x) for x in [0.1, 0.2, 0.3, 0.4, 0.5]])
    plt.savefig('./figures/probabilistic_guarantee.pdf', dpi=300, bbox_inches='tight')


def alpha_against_m():
    # x = np.linspace(0.4, 0.8, 50)
    x = np.linspace(0.5, 0.999, 20)
    # x = [0.5, 0.6, 0.7, 0.8, 0.9, 0.999]

    sns.set_style('white')
    plt.figure(figsize=(4.5, 4.5))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)

    # y = [[math.ceil(math.log(1 - alpha, 1 - xx)) for xx in x] for alpha in [0.70, 0.80, 0.85, 0.9, 0.99]]
    # plt.stackplot(x, y, labels=[f'$\\alpha$ = {alpha:.2f}' for alpha in [0.70, 0.80, 0.85, 0.9, 0.99]], baseline='zero')
    # for alpha in [0.70, 0.80, 0.85, 0.9, 0.99]:
    for alpha in [0.70, 0.9, 0.99]:
        y = [math.ceil(math.log(1 - alpha, 1 - xx)) for xx in x]

        # ax = sns.scatterplot(x=x, y=y, label=f'$\\alpha$ = {alpha:.2f}', legend=True, marker='o')
        plt.step(x, y, where='mid', label=f'$\\alpha$ = {alpha:.2f}')
        # ax.set_yticklabels([f'{math.ceil(x)}' for x in ax.get_yticks()])
    # plt.xlabel('Soundness/completeness probability (p/q)')
    # plt.xlabel('Soundness probability (p)')
    plt.xlabel('Necessity probability ($p_N$)')
    plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.ylabel('Lowerbound')
    plt.legend()
    # plt.yticks([1, 2])
    # ax.set_xticklabels(['{:,.0%}'.format(x) for x in [0.1, 0.2, 0.3, 0.4, 0.5]])
    # plt.show()
    plt.tight_layout()
    plt.savefig('./figures/alpha_against_m.pdf', dpi=300, bbox_inches='tight')

def beta_against_m():
    # x = np.linspace(0.4, 0.8, 100)
    x = np.linspace(0.5, 0.99, 20)

    sns.set_style('white')
    plt.figure(figsize=(4.75, 4.5))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    # for beta in [0.70, 0.80, 0.85, 0.9, 0.99]:
    for beta in [0.70, 0.9, 0.99]:
        y = [math.floor(math.log(beta, xx)) for xx in x]

        # sns.lineplot(x=x, y=y, label=f'{k:.0%} as challenge edges', legend=True)
        plt.step(x, y, where='mid', label=f'$\\beta$ = {beta:.2f}')
        # ax = sns.lineplot(x=x, y=y, label=f'$\\beta$ = {beta:.2f}', legend=True)
        # ax.set_yticklabels([f'{math.ceil(x)}' for x in ax.get_yticks()])
    # plt.xlabel('Soundness/completeness probability (p/q)')
    plt.xlabel('Soundness probability ($p_S$)')
    plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # plt.xlabel('Completeness probability (q)')
    plt.ylabel('Upperbound')
    plt.legend()
    # plt.yticks([1, 2])
    # ax.set_xticklabels(['{:,.0%}'.format(x) for x in [0.1, 0.2, 0.3, 0.4, 0.5]])
    # plt.show()
    plt.tight_layout()
    plt.savefig('./figures/beta_against_m.pdf', dpi=300, bbox_inches='tight')



def mixing_verify(result_filename):
    df = pd.read_csv(result_filename)
    # df['tpr'] = df['tp'] / (df['tp'] + df['fn'])
    # df['tnr'] = df['tn'] / (df['tn'] + df['fp'])
    df = df.pivot(index='m', columns='k', values='tpr')
    print(df)

    sns.set_style('whitegrid')
    plt.figure(figsize=(11, 9))
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    # c = sns.color_palette('Set2', n_colors=3)
    # f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # mask = np.diag(np.diag(np.ones((5,5), dtype=bool)))

    # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(df, mask=mask, cmap=cmap, center=0,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    sns.heatmap(df, cmap=cmap, center=0.95)
    # labels = plt.gca().get_legend_handles_labels()

    # plt.xlabel('Search space M')
    # plt.xticks([1, 2, 3, 4, 5])

    # plt.xlabel('p')
    # plt.ylabel('TPR/TNR')
    # plt.legend([], [], frameon=False)
    plt.savefig(f'./figures/mixing_verify2.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    # ax_legend.axis('off')
    # legend_names = ['TPR', 'TNR']
    # ax_legend.legend(labels[0], legend_names, loc='center', ncol=3)
    # fig_legend.savefig('./figures/mixing_verify_legend.pdf', dpi=300, bbox_inches='tight')

def empirical_vs_theortical(result_filename):
    df = pd.read_csv(result_filename)

    sns.set_style('white')
    plt.figure()
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)
    
    sns.lineplot(data=df, x='p', y='tpr')

    sns.lineplot(data=df, x='p', y='p', linestyle='--', color='red')
    plt.xlabel('P(A)')
    plt.ylabel('TPR')
    # plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    # ax.set_xticklabels(['{:,.0%}'.format(x) for x in [0.1, 0.2, 0.3, 0.4, 0.5]])
    # plt.savefig('./figures/probabilistic_guarantee.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def differnt_surrogates():
    df = pd.DataFrame({
        'P': [r'P_{TP}', r'P_{TN}', r'P_{TP}', r'P_{TN}', r'P_{TP}', r'P_{TN}', r'P_{TP}', r'P_{TN}', r'P_{TP}', r'P_{TN}'],
        'Effectiveness': [0.844, 1, 0.872, 1, 0.866, 1, 0.888, 1, 0.960, 1],
    })


    custom = {
        "axes.edgecolor": "gray", 
        # "grid.linestyle": "dashed", 
        "grid.color": "gray",
    } 
    sns.set_style("white", rc = custom)
    plt.figure()
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)

    # hue_order = ['ours', 'nettack', 'minmax']
    c = sns.color_palette('Set1', n_colors=7)
    ax = sns.lineplot(x='incomplete_ratio', y='tpr', data=df, marker="o",
                      errorbar=None, markers=True, dashes=False, markersize=10)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    labels = plt.gca().get_legend_handles_labels()
    print(labels)

    plt.xlabel('Incompleteness ratio')
    plt.xticks([0.2, 0.4, 0.6, 0.8])
    # plt.xlabel('p')
    # plt.ylim(0.4, 1.0)
    plt.ylabel('True positive rate')
    # plt.legend([], [], frameon=False)
    plt.savefig(f'./figures/incomplete_ratio.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def incomplete_ratio(result_filename):
    df = pd.read_csv(result_filename)

    custom = {
        "axes.edgecolor": "gray", 
        # "grid.linestyle": "dashed", 
        "grid.color": "gray",
    } 
    sns.set_style("white", rc = custom)
    plt.figure()
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)

    # hue_order = ['ours', 'nettack', 'minmax']
    c = sns.color_palette('Set1', n_colors=2)
    ax = sns.lineplot(x='incomplete_ratio', y='tpr', data=df, marker="o",
                      errorbar=None, markers=True, dashes=False, markersize=10)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    labels = plt.gca().get_legend_handles_labels()
    print(labels)

    plt.xlabel('Incompleteness ratio')
    plt.xticks([0.2, 0.4, 0.6, 0.8])
    # plt.xlabel('p')
    # plt.ylim(0.4, 1.0)
    plt.ylabel('True positive rate')
    # plt.legend([], [], frameon=False)
    plt.savefig(f'./figures/incomplete_ratio.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
    # ax_legend.axis('off')
    # legend_names = ['IVEU (ours)', 'Nettack', 'IG-Attack', 'SGAttack', 'FGA', 'RND']
    # ax_legend.legend(labels[0], legend_names, loc='center', ncol=6)
    # fig_legend.savefig('./figures/main_legend.pdf', dpi=300, bbox_inches='tight')


def incomplete_ratio_tpr_tnr(result_filename):
    df = pd.read_csv(result_filename)

    custom = {
        "axes.edgecolor": "gray", 
        # "grid.linestyle": "dashed", 
        "grid.color": "gray",
    } 
    sns.set_style("white", rc = custom)
    plt.rc('axes', labelsize=24)
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=24)

    fig, ax1 = plt.subplots()
    c = sns.color_palette('Set1', n_colors=2)
    ax1 = sns.lineplot(x='incomplete_ratio', y='tpr', data=df, marker='o', markers=True, dashes=False, markersize=10, color=c[0], label='TPR')
    ax1.set_ylabel(r'$P_{TP}$')
    ax1.set_xlabel('Incompleteness ratio')
    ax1.tick_params(axis='y')
    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax1.legend([], [], frameon=False)

    ax2 = ax1.twinx()
    sns.lineplot(x='incomplete_ratio', y='tnr', data=df, marker='X', markers=True, dashes=False, markersize=10, color=c[1], label='TNR', ax=ax2)
    ax2.set_ylabel(r'$P_{TN}$')
    ax2.tick_params(axis='y')
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax2.legend([], [], frameon=False)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # plt.legend([], [], frameon=False)
    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    plt.xlabel('Incompleteness ratio')
    plt.savefig(f'./figures/incomplete_ratio_tpr_tnr_citeseer.pdf', dpi=300, bbox_inches='tight')

    fig_legend, ax_legend = plt.subplots(figsize=(5, 1))
    ax_legend.axis('off')
    legend_names = [r'$P_{TP}$', r'$P_{TN}$']
    ax_legend.legend(lines_1 + lines_2, legend_names, loc='center', ncol=2)
    fig_legend.savefig('./figures/incomplete_ratio_tpr_tnr_legend.pdf', dpi=300, bbox_inches='tight')


def num_challenge_edges(result_filename):
    df = pd.read_csv(result_filename)
    df = df[df['num_pert_edges'] < 25]

    custom = {
        "axes.edgecolor": "gray", 
        # "grid.linestyle": "dashed", 
        "grid.color": "gray",
    } 
    sns.set_style("white", rc = custom)
    plt.figure()
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('legend', fontsize=16)
    plt.rc('font', size=20)

    # hue_order = ['ours', 'nettack', 'minmax']
    c = sns.color_palette('Set1', n_colors=7)
    ax = sns.lineplot(x='num_pert_edges', y='tpr', data=df, marker="o",
                      errorbar=None, markers=True, dashes=False, markersize=10)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    labels = plt.gca().get_legend_handles_labels()
    print(labels)

    plt.xlabel('Number of challenge edges')
    plt.xticks([5, 10, 15, 20])
    # plt.xlabel('p')
    # plt.ylim(0.4, 1.0)
    plt.ylabel('True positive rate')
    # plt.legend([], [], frameon=False)
    plt.savefig(f'./figures/num_challenge_edges.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # boundary_score_vs_asr(sys.argv[1])
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--result', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--type', type=str, default=None)
    args = parser.parse_args()

    print(args.task)

    if args.task == 'main':
        assert args.dataset is not None, 'Please specify the dataset'
        main_result(args.result, args.dataset)
    elif args.task == 'efficiency':
        assert args.type is not None, 'Please specify the type of efficiency'
        efficiency(args.result, args.type)
    elif args.task == 'sensitivity':
        assert args.type is not None, 'Please specify the type of sensitivity'
        sensitivity(args.result, args.type)
    elif args.task == 'guarantee':
        probabilistic_guarantee()
    elif args.task == 'mixing':
        mixing_verify(args.result)
    elif args.task == 'empirical':
        empirical_vs_theortical(args.result)
    elif args.task == 'm':
        alpha_against_m()
        beta_against_m()
    elif args.task == 'incomplete_ratio':
        # incomplete_ratio_result(args.result, 'citeseer')
        incomplete_ratio_tpr_tnr(args.result)
    elif args.task == 'num_challenge_edges':
        num_pert_edges_result(args.result, 'citeseer')