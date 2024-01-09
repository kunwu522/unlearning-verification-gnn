import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from torch_geometric.utils import k_hop_subgraph

import argument
import data_loader


if __name__ == '__main__':
    parser = argument.load_parser()
    args = parser.parse_args()

    data = data_loader.load(args)
    # filename = 'fragile_cvx_cora_label1_2hop_all20_1700451181.json'
    # filename = 'fragile_cvx_citeseer_label1_2hop_all20_1700451493.json'
    filename = 'citeseer1.json'

    with open(os.path.join('./', filename), 'r') as fp:
        dict_result = json.load(fp)

    result = defaultdict(list)
    for trial, trial_data in dict_result.items():
        for candidate_result in trial_data:
            target_node = candidate_result['target_node']
            result['target_node'].append(target_node)
            result['ground_truth'].append(candidate_result['ground_truth'])
            result['prediction'].append(candidate_result['prediction'])
            attack_label = candidate_result['second_best_label']
            result['attack_label'].append(attack_label)
            if candidate_result['certified'][str(attack_label)]['fragile'] and candidate_result['certified'][str(attack_label)]['single_predictions'][0] == attack_label:
                r = 'TP'
            elif candidate_result['certified'][str(attack_label)]['fragile'] and candidate_result['certified'][str(attack_label)]['single_predictions'][0] != attack_label:
                r = 'FP'
            elif not candidate_result['certified'][str(attack_label)]['fragile'] and candidate_result['certified'][str(attack_label)]['single_predictions'][0] == attack_label:
                r = 'FN'
            else:
                r = 'TN'
            result['result'].append(r)

            subset, edge_index, mapping, edge_mask = k_hop_subgraph(target_node, 1, data.edge_index)
            target_1hop_dist = np.zeros((data.num_classes), dtype=np.int32)
            for u in subset:
                if u == target_node:
                    continue
                target_1hop_dist[data.y[u]] += 1
             
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(target_node, 2, data.edge_index)
            target_2hop_dist = np.zeros((data.num_classes), dtype=np.int32)
            for u in subset:
                if u == target_node:
                    continue
                target_2hop_dist[data.y[u]] += 1
            result['target deg.'].append(data.degree(target_node))
            result['target 1hop dist.'].append(str(target_1hop_dist))
            result['target 2hop dist.'].append(str(target_2hop_dist))

            perturbation = candidate_result['certified'][str(attack_label)]['perturbations'][0][1]
            pert_degree = data.degree(perturbation)
            subset, _, _, _ = k_hop_subgraph(perturbation, 1, data.edge_index)
            pert_1hop_dist = np.zeros((data.num_classes), dtype=np.int32)
            for u in subset:
                if u == perturbation:
                    continue
                pert_1hop_dist[data.y[u]] += 1
            subset, _, _, _ = k_hop_subgraph(perturbation, 2, data.edge_index)
            pert_2hop_dist = np.zeros((data.num_classes), dtype=np.int32)
            for u in subset:
                if u == perturbation:
                    continue
                pert_2hop_dist[data.y[u]] += 1

            result['pert. deg.'].append(data.degree(pert_degree))
            result['pert. 1hop dist.'].append(str(pert_1hop_dist))
            result['pert. 2hop dist.'].append(str(pert_2hop_dist))
    
    df = pd.DataFrame(result)
    df.to_csv(f'{filename}.csv')