from tqdm import tqdm
import numpy as np
from scipy.stats import entropy
import torch
from torch_geometric.utils import k_hop_subgraph
import argument
import data_loader
import utils


if __name__ == '__main__':
    parser = argument.load_parser()
    args = parser.parse_args()

    data = data_loader.load(args)
    candidate_nodes = data.train_set.nodes.tolist()

    node2homo_by_modularity = {}
    node2homo_by_entropy = {}
    for v in tqdm(candidate_nodes):
        subgraph, _edge_index, _, _ = k_hop_subgraph(v, 2, data.edge_index, relabel_nodes=True)
        _direct_edge_index = utils.to_directed(_edge_index)
        node2homo_by_modularity[v] = utils.homophily(subgraph.tolist(), _direct_edge_index.t().tolist(), data.y[subgraph].tolist())
        # homo2 = utils.homophily_modularity(subgraph.tolist(), _direct_edge_index.t().tolist(), data.y[subgraph].tolist())

        # labels = data.y[subgraph]
        # label_dist = np.zeros((data.num_classes))
        # for label, count in zip(*np.unique(labels, return_counts=True)):
        #     label_dist[label] = count
        # label_dist /= np.sum(label_dist)
        # homo = entropy(label_dist)
        # node2homo_by_entropy[v] = entropy(label_dist)
    

    sorted_node2homo_by_modularity = {k: v for k,v in sorted(node2homo_by_modularity.items(), key=lambda item: item[1])}
    top100_modularity = list(sorted_node2homo_by_modularity.keys())[:100]
    modularity_values = list(sorted_node2homo_by_modularity.values())
    top100_modularity_values = modularity_values[:100]
    sorted_node2homo_by_entropy = {k: v for k,v in sorted(node2homo_by_entropy.items(), key=lambda item: item[1], reverse=True)}
    top100_entropy = list(sorted_node2homo_by_entropy.keys())[:100]

    print('overlap of two methods:', len(set(top100_modularity).intersection(top100_entropy)))
    
    print('Max:', np.max(modularity_values))
    print('Min:', np.min(modularity_values))
    print('Avg:', np.mean(modularity_values))
    print('-' * 50)
    print('Max:', np.max(top100_modularity_values))
    print('Min:', np.min(top100_modularity_values))
    print('Avg:', np.mean(top100_modularity_values))
    print(top100_modularity_values)