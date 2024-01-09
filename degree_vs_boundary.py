import numpy as np
from torch_geometric.utils import k_hop_subgraph
import argument
import data_loader
from model.gcn import GNN
import utils


if __name__ == '__main__':
    parser = argument.load_parser()
    args = parser.parse_args()

    data = data_loader.load(args)
    device = utils.get_device(args)

    model = GNN(args, data.num_features, data.num_classes, surrogate=False)
    model.train(data, device)

    candidates = data.train_set.nodes.tolist()
    predictions, labels, posteriors = model.predict(data, device, target_nodes=candidates, return_posterior=True)

    node2degree = {}
    for v in data.train_set.nodes.tolist():
        node2degree[v] = data.degree(v)

    node2boundary_score = {}
    for v, post in zip(candidates, posteriors):
        node2boundary_score[v] = utils.boundary_score(post)


    node2homophily = {}
    for v in candidates:
        if data.degree(v) == 0:
            continue

        subgraph, _edge_index, _, _ = k_hop_subgraph(v, 2, data.edge_index, relabel_nodes=True)
        node2homophily[v] = utils.homophily(
            subgraph.tolist(), 
            utils.to_directed(_edge_index).t().tolist(),
            data.y[subgraph].tolist()
        )
        # node2homophily[v] = utils.homophily_entropy(data.num_classes, data.y[subgraph])

    sorted_node2homophily = {k: v for k,v in sorted(node2homophily.items(), key=lambda item: item[1])}
    sorted_node2degree = {k: v for k, v in sorted(node2degree.items(), key=lambda item: item[1], reverse=True)}
    sorted_node2boundary_score = {k: v for k, v in sorted(node2boundary_score.items(), key=lambda item: item[1], reverse=True)}

    print('The top 20 largest degree nodes')
    print(' ', list(sorted_node2degree.keys())[:20])
    print('-' * 40)
    print('The top 20 largest degree')
    print(' ', list(sorted_node2degree.values())[:20])
    print('=' * 40)
    print('The top 20 smallest degree nodes')
    print(' ', list(sorted_node2degree.keys())[::-1][:20])
    print('-' * 40)
    print('The top 20 smallest degree')
    print(' ', list(sorted_node2degree.values())[::-1][:20])
    print()

    print('The top 20 boundary nodes')
    boundary_nodes = list(sorted_node2boundary_score.keys())[::-1][:20]
    print(' ', boundary_nodes)
    print(' ', 'degree', [node2degree[v] for v in boundary_nodes], ', mean:',np.mean([node2degree[v] for v in boundary_nodes]))
    print(' ', 'label', np.unique(data.y[boundary_nodes].tolist(), return_counts=True))
    print('-' * 40)
    print('The top 20 distant nodes')
    distant_nodes = list(sorted_node2boundary_score.keys())[:20]
    print(' ', distant_nodes)
    print(' ', 'degree', [node2degree[v] for v in boundary_nodes], ', mean:',np.mean([node2degree[v] for v in distant_nodes]))
    print(' ', 'label', np.unique(data.y[distant_nodes].tolist(), return_counts=True))
    print('-' * 40)
    homo_nodes = list(sorted_node2homophily.keys())[:20]
    print(' ', homo_nodes)
    print(' ', 'degree', [node2degree[v] for v in boundary_nodes], ', mean:',np.mean([node2degree[v] for v in homo_nodes]))
    print(' ', 'label', np.unique(data.y[homo_nodes].tolist(), return_counts=True))
    print('-' * 40)