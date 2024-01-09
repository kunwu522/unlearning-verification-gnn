import torch
import numpy as np
from collections import defaultdict
from torch_geometric.utils import k_hop_subgraph
from scipy.spatial import distance
import argument
import data_loader
from model.gcn import GNN
import utils


def analyze_dist(args):
    device = utils.get_device(args)
    data = data_loader.load(args)

    model = GNN(args, data.num_features, data.num_classes, surrogate=False)
    model.train(data, device)

    nodes = list(range(data.num_nodes))
    predictions, labels = model.predict(data, device, target_nodes=nodes)

    dist2pred = {}
    dist2truth = {}
    for v, pred, label in zip(nodes, predictions, labels):
        subset, _, _, _ = k_hop_subgraph(v, 1, data.edge_index)
        _dist = np.zeros((data.num_classes))
        _dist[label] += 1
        for u in subset:
            _label = data.y[u].item()
            _dist[_label] += 1
        key = tuple(_dist.tolist())
        if key not in dist2pred:
            dist2pred[key] = defaultdict(int)
        dist2pred[key][pred] += 1
        if key not in dist2truth:
            dist2truth[key] = defaultdict(int)
        dist2truth[key][label] += 1
    
    num_dist = len(dist2pred)
    print(f'There are {num_dist} diff. distributions.')
    # print(dist2pred)

    inunique_dist = {}
    for k, v in dist2pred.items():
        if len(v) > 1:
            inunique_dist[k] = v
    print(f'Inunique {len(inunique_dist)} prediction dist:', dict(inunique_dist)) 
    inunique_dist = {}
    for k, v in dist2truth.items():
        if len(v) > 1:
            inunique_dist[k] = v
    print(f'Inunique {len(inunique_dist)} prediction dist:', dict(inunique_dist)) 


def analyze_pairs_in_dist_prediction(args):
    device = utils.get_device(args)
    data = data_loader.load(args)

    model = GNN(args, data.num_features, data.num_classes, surrogate=False)
    model.train(data, device)

    test_nodes = data.test_set.nodes.tolist()

    node2neighbor_dist = {}
    for v in test_nodes:
        subset, edge_index, _, _ = k_hop_subgraph(v, 2, data.edge_index)
        # _dist = defaultdict(int)
        _dist = np.zeros((data.num_classes))
        for u in subset:
            _label = data.y[u].item()
            _dist[_label] += 1
        
        node2neighbor_dist[v] = _dist / np.sum(_dist)

    result = defaultdict(list)
    pred, truth = model.predict(data, device, target_nodes=test_nodes)
    for i in range(len(test_nodes)):
        for j in range(i+1, len(test_nodes)):
            u, v =  test_nodes[i], test_nodes[j]
            pred_u, pred_v = pred[i], pred[j]
            label_u, label_v = truth[i], truth[j]
            dist_u, dist_v = node2neighbor_dist[u], node2neighbor_dist[v]
            js_distance = distance.jensenshannon(dist_u, dist_v)
            # print(dist_u, dist_v, js_distance)
            # exit(0)

            result['pair'].append((u, v))
            result['prediction'].append((pred_u, pred_v))
            result['label'].append((label_u, label_v))
            result['distance'].append(js_distance)

    sorted_indices = np.argsort(result['distance'])
    print(sorted_indices)
    print('The top 50 pairs')
    print('      nodes:', np.array(result['pair'])[sorted_indices[:30]].tolist()) 
    print('predictions:', np.array(result['prediction'])[sorted_indices[:30]].tolist()) 
    print('     labels:', np.array(result['label'])[sorted_indices[:30]].tolist()) 
    print('   distance:', np.array(result['distance'])[sorted_indices[:30]].tolist())



if __name__ == '__main__':
    parser = argument.load_parser()
    args = parser.parse_args()
    # analyze_pairs_in_dist_prediction(args)
    analyze_dist(args)



