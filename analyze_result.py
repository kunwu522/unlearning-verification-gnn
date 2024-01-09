import sys
import math
import numpy as np
import pandas as pd
import utils

def calc_q(node_tokens, df):
    node_tokens2q = {}
    num_unsuccessful_attack = 0
    for v in node_tokens:
        _df = df[df['target node'] == v]
        attack_success_df = _df[_df['adv prediction'] != _df['clean prediction']]
        if len(attack_success_df) > 0:
            q = np.sum(attack_success_df['# of flip'] > 0) / len(attack_success_df)
            node_tokens2q[v] = q
        else:
            node_tokens2q[v] = 0
            num_unsuccessful_attack += 1

    print('num_unsuccessful_attack', num_unsuccessful_attack)
    return node_tokens2q 


if __name__ == '__main__':
    result_path1 = sys.argv[1]
    # print('Analyzing', result_path1, result_path2)

    df = pd.read_csv(result_path1)
    node_tokens = pd.unique(df['target node'])
    token2q = calc_q(node_tokens, df)
    token2asr = utils.calc_asr_by_tokens(df)
    asr1 = np.mean(list(token2asr.values()))
    print(token2q)
    # exit(0)

    mean_q = np.mean(list(token2q.values()))
    max_q = np.max(list(token2q.values()))
    num_trials = 10

    if len(sys.argv) > 2:
        result_path2 = sys.argv[2]
        df = pd.read_csv(result_path2)
        # num_node_tokens = len(df)
        # asr = np.sum(df['adv prediction'] != df['clean prediction']) / num_node_tokens
        # k = np.sum((df['adv prediction'] != df['clean prediction']) & (df['# of flip'] == 0))
        # mean_p = math.comb(num_node_tokens, k) * (mean_q ** k) * ((1 - mean_q) ** (num_node_tokens - k))
        # max_p = math.comb(num_node_tokens, k) * (max_q ** k) * ((1 - max_q) ** (num_node_tokens - k))
        mean_p = utils.calc_removal_probability(df, mean_q, num_trials)
        max_p = utils.calc_removal_probability(df, max_q, num_trials)
        asr2 = utils.calc_asr_by_trials(df, num_trials)

        print('Using mean q')
        print(f'  ==> q: {mean_q:.4f}')
        print(f'  ==> asr: {asr1:.4f}/{asr2:.4f}')
        # print(f'  ==> k: {k}')
        print(f'  ==> p: {mean_p:.8f}')
        print('-' * 30)
        
        print('Using max q')
        print(f'  ==> q: {max_q:.4f}')
        print(f'  ==> asr: {asr1:.4f}/{asr2:.4f}')
        # print(f'  ==> k: {k}')
        print(f'  ==> p: {max_p:.8f}')
        print('-' * 30)
    else:
        print(f'  ==> q: {mean_q:.4f}')
        print(f'  ==> asr: {asr1:.4f}')
        


    # q1s, q2s,  = [], []
    # asr = []
    # num_edge_tokens = []
    # step = 20
    # for i in range(0, len(df), step):
    #     _df = df[i: i+step]
    #     q1 = np.sum(_df['# of flip']>0) / len(_df)
    #     q1s.append(q1)

    #     _df['# of ']
        
    #     __df = _df[_df['adv prediction'] != _df['clean prediction']]
    #     asr.append(len(__df) / len(_df))
    #     if len(__df) > 0:
    #         q2 = np.sum(__df['# of flip']>0) / len(__df)
    #         q2s.append(q2)
    #     else:
    #         q2s.append(0)

    # print('The avg q :', q1s, np.mean(q1s))
    # print('The asr: ', asr, np.mean(asr)) 
    # print('The avg q (consider K_vt):', q2s, np.mean(q2s))
    