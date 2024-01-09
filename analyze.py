import os
import pickle
import argparse
from collections import defaultdict
from ast import literal_eval
import numpy as np
import pandas as pd

import utils


def analyze_boundary_vs_fragile(df, boundary_nodes_file):
    with open(boundary_nodes_file, 'rb') as f:
        boundary_nodes = pickle.load(f)
    
    fragile_nodes = df['target'].values
    boundary_nodes = list(boundary_nodes.keys())

    for i in range(0, 300, 50):
        print(f'{i} - {i + 50} contains:', len(set(boundary_nodes[i:i+50]) & set(fragile_nodes)))


def analyze_fragile_results(df):
    statistic = defaultdict(int)
    for idx, row in df.iterrows():
        single_adv_preds = literal_eval(row['single adv predictions'])
        clean_pred = row['clean prediction']
        if len(single_adv_preds) == 0:
            statistic['fail attack'] += 1
        for single_pred in single_adv_preds:
            if clean_pred != single_pred:
                statistic['fragile'] += 1
                break

    print('Result ==>')
    print('  total:', len(df))
    print('  result:', statistic)
    print('-' * 50)


def analyze_stability(result_file):
    df = pd.read_csv(os.path.join('result', result_file))
    num_fragile_nodes = []
    fragileness = []
    consistency = []

    # tmp = [0, 63, 51, 55, 54, 64] # Citeseer N=10000
    # tmp = [0, 55, 54, 48, 50, 61] # Citeseer N=1000
    # tmp = [0, 60, 54, 48, 59, 49] # Citeseer N=100
    # tmp = [0, 34, 38, 37, 23, 26] # Cora N=100
    # tmp = [0, 42, 33, 35, 44, 32] # Cora N=1000
    # tmp = [0, 40, 39, 45, 48, 39] # Cora N=10000
    # for i in range(len(tmp) - 1):
    for t in range(5):
        _df = df[df['trial'] == t]
        # _df = df[sum(tmp[:i+1]): sum(tmp[:i+2])]
        num_fraigle = _df['fragileness'].values.sum()
        num_consistent = _df[_df['fragileness'] == 1]['consistency'].values.sum()
        num_fragile_nodes.append(len(_df))
        fragileness.append(num_fraigle / len(_df))
        consistency.append(num_consistent / num_fraigle)
    print('Result ==>')
    print('# of fragile nodes:', num_fragile_nodes, np.mean(num_fragile_nodes), np.std(num_fragile_nodes))
    print('fragilenss:', fragileness, np.mean(fragileness), np.std(fragileness))
    print('consistency:', consistency, np.mean(consistency), np.std(consistency))


def analyze_baseline(result_file):
    df = pd.read_csv(os.path.join('result', result_file))
    fpr = utils.calc_fpr(df, 100)
    fnr = utils.calc_fnr(df, 100)
    print('-' * 50, 'Baseline Result', '-' * 50)
    print('  ==> FPR:', fpr)
    print('  ==> FNR:', fnr)
    print('=' * 100)


def analyze_main_result(result_file):
    df = pd.read_csv(os.path.join('result', result_file))
    tpr_result, tnr_result = {}, {}
    for method in ['ours', 'nettack', 'minmax']:
        tpr_result[method], tnr_result[method] = defaultdict(list), defaultdict(list)
    for t in range(3):
        _df_trial = df[df['trial'] == t]
        for method in ['ours', 'nettack', 'minmax']:
            for num_nodes in [10, 20, 30, 40, 50]:
                _df = _df_trial[_df_trial['method'] == method][:num_nodes]
                _df = _df[~(_df['perturbations'] == '[]')]
                tpr_result[method][num_nodes].append((_df['prediction'] != _df['adv prediction']).sum() / len(_df))
                tnr_result[method][num_nodes].append((len(_df) - _df['incompleteness'].sum()) / len(_df))

        # tpr.append(utils.calc_tpr(_df, 100))
        # tnr.append(utils.calc_tnr(_df, 100))
    
    for method in ['ours', 'nettack', 'minmax']:
        for num_nodes in [10, 20, 30, 40, 50]:
            print(f'  ==> {method} TPR@{num_nodes}: {np.mean(tpr_result[method][num_nodes])*100:.1f}% ± {np.std(tpr_result[method][num_nodes])*100:.1f}%.')
            print(f'  ==> {method} TNR@{num_nodes}: {np.mean(tnr_result[method][num_nodes])*100:.1f}% ± {np.std(tnr_result[method][num_nodes])*100:.1f}%.')

def analyze_main_result2(result_file):
    df = pd.read_csv(os.path.join('result', result_file))
    methods = pd.unique(df['method'])
    num_node_tokens = pd.unique(df['num node tokens'])
    ps = pd.unique(df['p'])
    for method in methods:
        for num_node_token in num_node_tokens:
            for p in ps:
                _df = df[(df['method'] == method) & (df['num node tokens'] == num_node_token) & (df['p'] == p)]
                tpr, tnr, fpr, fnr = [], [], [], []
                for index, row in _df.iterrows():
                    num_trial = int(row['tp']) + int(row['fp'])
                    tpr.append(int(row['tp']) / num_trial)
                    tnr.append(int(row['tn']) / num_trial)
                    fpr.append(int(row['fp']) / num_trial)
                    fnr.append(int(row['fn']) / num_trial)
                print(f'  ==> {method} TPR p={p}: {np.mean(tpr)*100:.1f}% ± {np.std(tpr)*100:.1f}%.')
                print(f'  ==> {method} TNR p={p}: {np.mean(tnr)*100:.1f}% ± {np.std(tnr)*100:.1f}%.')
                print(f'  ==> {method} FPR p={p}: {np.mean(fpr)*100:.1f}% ± {np.std(fpr)*100:.1f}%.')
                print(f'  ==> {method} FNR p={p}: {np.mean(fnr)*100:.1f}% ± {np.std(fnr)*100:.1f}%.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--result', type=str, required=True)
    args = parser.parse_args()

    if args.task == 'fragile':
        analyze_fragile_results(pd.read_csv(args.result))
    elif args.task == 'boundary':
        # analyze_boundary_vs_fragile(pd.read_csv(args.result), './result/cora/boundary_nodes_1694119708.pkl')
        # analyze_boundary_vs_fragile(pd.read_csv(args.result), './result/cora/boundary_nodes_1694196992.pkl')
        analyze_boundary_vs_fragile(pd.read_csv(args.result), './result/cora/boundary_nodes_1694204358.pkl')
    elif args.task == 'stability':
        analyze_stability(args.result)
    elif args.task == 'table2':
        analyze_baseline(args.result)
    elif args.task == 'main':
        # analyze_main_result(args.result)
        analyze_main_result2(args.result)
    else:
        pass

    a = 0.3667755102040816
    b = 0.020550686011506378

    print(f'{a * 100:.1f}% ± {b*100:.1f}%')

    
