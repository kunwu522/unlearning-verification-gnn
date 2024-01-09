import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph, is_undirected

if __name__ == '__main__':
    nodes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    edges = [(0, 1), (1, 0), (0, 9), (9, 0), (1, 2), (2, 1), (1, 7), (7, 1),
             (2, 3), (3, 2), (4, 5), (5, 4), (5, 6), (6, 5), (5, 7), (7, 5),
             (7, 8), (8, 7), (4, 10), (10, 4), (6, 11), (11, 6), (4, 12), (12, 4),
             (10, 13), (13, 10)]
    labels = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
    features = np.random.rand(14, 5)

    edge_index = torch.tensor(edges).t()
    print('Undirected graph:', is_undirected(edge_index))

    target_node = 7
    k_hop_nodes, _, _, _ = k_hop_subgraph(target_node, 2, edge_index)
    print('k_hop_nodes:', k_hop_nodes)

    independent_nodes = nodes[~torch.isin(nodes, k_hop_nodes)]
    print('independent nodes:', independent_nodes)