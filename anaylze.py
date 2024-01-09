import os
import sys
import numpy as np
import pandas as pd

if __name__ == '__main__':
    print('Analyze ', sys.argv[1])

    df = pd.read_csv(os.path.join('./result', sys.argv[1]))
    # clean_pred = df['clean prediction'].values
    # adv_pred = df['adv prediction'].values

    # asr = np.sum(clean_pred != adv_pred) / len(clean_pred)
    # print('-' * 10 + 'ASR' + '-' * 10)
    # print(' ' * 10, f'{asr:.4f}')
    # print('=' * 25)

    # num_adv_edges = df['num adv edges'].values

    # print('-' * 5 + 'Statistic of # of adv edges' + '-' * 5)
    # print('  Totoal number  ==> ', np.sum(num_adv_edges))
    # print('  Average number ==> ', np.mean(num_adv_edges))
    # print('=' * 20)


    num_overlap = 0
    for i in range(10):
        adv_nodes = set()
        adv_edges = df['num adv edges'].values[i * 100: (i+1) * 100]
        target_nodes = df['target node'].values[i * 100: (i+1) * 100]

        for target_node, adv_edges in zip(target_nodes, adv_edges):
            _adv_nodes = adv_edges[1:-1]
            _adv_nodes = _adv_nodes.replace('[', '')
            _adv_nodes = _adv_nodes.replace(']', '')
            _adv_nodes = _adv_nodes.split(',')
            for n in _adv_nodes:
                if int(n) == target_node:
                    continue
                adv_nodes.add(int(n))

        num_overlap += len(set(target_nodes).intersection(adv_nodes)) 
        print('The number of nodes targets that appear in adv nodes:', num_overlap / (i+1))



