import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score
from graph_backdoor.trojan import train_gtn, GraphTrojanNet
from graph_backdoor.bkdcdd import backdoor_data

import argument
import data_loader
import utils

def sample_subgraph(args, data, num, size):
    _nodes = np.arange(data.num_nodes)
    _edges = data.edge_index.t().numpy()
    subgraphs = []
    for _ in range(num):
        subnodes = np.random.choice(_nodes, size, replace=False)
        _edges1 = _edges[np.where(np.in1d(_edges[:, 0], subnodes))]
        _edges2 = _edges[np.where(np.in1d(_edges[:, 1], subnodes))]
        subedges = np.concatenate([_edges1, _edges2], axis=0)
        subgraphs.append((subnodes, subedges))

    return subgraphs

def init_triggers(args, data, subgraphs, target_label):
    _data = copy.deepcopy(data)
    _edges = _data.edge_index.t().numpy()
    trigger_nodes = set()
    for _nodes, _ in subgraphs:
        trigger_nodes = trigger_nodes.union(set(_nodes))
        for idx, v1 in enumerate(_nodes):
            _data.x[v1] = 0.
            for v2 in _nodes[idx + 1:]:
                _edges = _edges[~np.all(_edges == [v1, v2], axis=1)]
                _edges = _edges[~np.all(_edges == [v2, v1], axis=1)]

    # change node labels
    nodes_bkd = _data.neighbors(list(trigger_nodes), 2)
    trigger_nodes = list(trigger_nodes.union(set(nodes_bkd)))
    _data.assign_labels(trigger_nodes, target_label)

    # change edges
    _data.edges = _edges

    return _data, trigger_nodes


def match_trigger(graph, trigger):
    pass


def assign_target_label(data, matchers, target_label, k):
    pass


def backdoor_success_rate(model, nodes, labels):
    pass


def generate_topo_input(args, data, device):
    # edges = data.edges
    # if len(args.hidden) == 0:
    #     i = torch.tensor(edges)
    #     v = torch.ones(len(edges))
    #     adj = torch.sparse_coo_tensor(list(zip(*i)), v, (num_nodes, num_nodes), device=device).to_dense()
    #     feat = torch.tensor(data.features, device=device)
    # elif len(args.hidden) == 1:

    adj = data.adjacency_matrix().to_dense().to(device)
    # 2-hop
    adj = torch.add(adj, torch.mm(adj, adj))
    adj = torch.where(adj > 0, torch.tensor(1.0, requires_grad=True, device=device),
                        torch.tensor(0.0, requires_grad=True, device=device))
    adj.fill_diagonal_(0.0)

    x = data.x.to(device)
    feat = torch.mm(adj, x)
    return adj, feat


def generate_masks(data, subgraphs, device):
    num_nodes = data.num_nodes
    topomask = torch.zeros(num_nodes, num_nodes).to(device)
    featmask = torch.zeros(num_nodes, data.x.shape[1]).to(device)
    for _nodes, _edges in subgraphs:
        for idx, v1 in enumerate(_nodes):
            featmask[v1][::] = 1
            for v2 in _nodes[idx + 1:]:
                topomask[v1, v2] = 1
                topomask[v2, v1] = 1
    return topomask, featmask


def generate_pos_and_neg(data, trigger_nodes):
    pos_train, neg_train = [], []
    for v in data.train_set.nodes:
        if v in trigger_nodes:
            pos_train.append(v)
        else:
            neg_train.append(v)
    return pos_train, neg_train


# def continue_train(self, args, data, pos_set, neg_set, device):
#     # edge_index = torch.cat(torch.where(bkd_adj > 0)).view(2, -1).to(device)
#     edge_index = torch.tensor(data.edges, device=device).t()
#     pos_label = torch.tensor(data.labels[pos_set], device=device)
#     neg_label = torch.tensor(data.labels[neg_set], device=device)

#     # model.update_embedding(torch.from_numpy(data.features).to(device))
#     optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.l2, betas=(0.5, 0.999))
#     scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)
#     loss_fn = F.cross_entropy

#     model.train()
#     for e in range(args.train_epochs):
#         optimizer.zero_grad()

#         losses = {'pos': 0.0, 'neg': 0.0}
#         output = model(pos_set, edge_index)
#         if len(output.shape) == 1:
#             output = output.unsqueeze(0)
#         losses['pos'] = loss_fn(output, pos_label)

#         output = model(neg_set, edge_index)
#         if len(output.shape) == 1:
#             output = output.unsqueeze(0)
#         losses['neg'] = loss_fn(output, neg_label)
#         loss = losses['pos'] + args.lambd * losses['neg']
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#     return model


def evaluate(model, data, trigger_nodes, device):
    edge_index = torch.tensor(data.edges, device=device).t()
    train_bkd_nodes = []
    test_bkd_nodes = []
    test_benign_nodes = []
    for v in data.train_set.nodes:
        if v in trigger_nodes:
            train_bkd_nodes.append(v)
    for v in data.test_set.nodes:
        if v in trigger_nodes:
            test_bkd_nodes.append(v)
        else:
            test_benign_nodes.append(v)
    test_benign_labels = data.y[test_benign_nodes]
    train_pred = model.predict(data, device, target_nodes=train_bkd_nodes)
    test_pred = model.predict(data, device, target_nodes=test_bkd_nodes)

    train_acc = accuracy_score(data.y[train_bkd_nodes], train_pred)
    test_acc = accuracy_score(data.y[test_bkd_nodes], test_pred)
    print(f'The ASR, training nodes: {train_acc:.4f}, test nodes: {test_acc:.4f}.')

    return train_acc, test_acc


# def convert_train_graph_to_transductive_graph(data):
#     nodes = data.train_set.nodes
#     node2idx = {v: idx for idx, v in enumerate(nodes)}
#     _features = data.features[nodes]
#     _labels = data.labels[nodes]
#     _edges = [(node2idx[v1], node2idx[v2]) for v1, v2 in data.train_edges]
#     _nodes = np.arange(len(nodes))
#     trans_graph = TransductiveGraph(data.name, _features, _nodes, _edges, _labels)
#     return trans_graph


# def inductive_backdoor_attack(args, model, data):
#     device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
#     bkd_data = backdoor_data(data, args.trigger_size)

#     node_nums = [d.num_nodes for d in data.dataset]
#     node_max = max(node_nums)
#     featdim = data.num_feat

#     topo_net = GraphTrojanNet(node_max)
#     feat_net = GraphTrojanNet(featdim)

#     for rs_step in range(10):
#         pass


def backdoor_attack(args, surrogate, data, only_addition=True):
    device = utils.get_device(args)
    l, c = np.unique(data.y, return_counts=True)
    target_label = l[np.argsort(c)[0]]
    print(f'Backdoor attack consider {target_label} as the target model.')

    subgraphs = sample_subgraph(args, data, args.num_triggers, args.trigger_size)
    trigger_data, trigger_nodes = init_triggers(args, data, subgraphs, target_label)
    bkd_data = copy.deepcopy(trigger_data)
    pos_set, neg_set = generate_pos_and_neg(data, trigger_nodes)

    topomask_train, featmask_train = generate_masks(data, subgraphs, device)
    topo_input, feat_input = generate_topo_input(args, trigger_data, device)

    # edges = data.edge_index.t().tolist()
    num_nodes = data.num_nodes
    # i = data.edge_index.t()
    # v = torch.ones(data.edge_index.size(1))
    # adj = torch.sparse_coo_tensor(list(zip(*i)), v, (num_nodes, num_nodes), device=device).to_dense()
    adj = data.adjacency_matrix().to(device)
    bkd_adj = copy.deepcopy(adj)

    # model_origin = train_model(args, trans_data, eval=False, verbose=False, device=device)
    # model_origin = GNN(args, data.num_features, data.num_classes, surrogate=False)
    # model_origin.train(data, device)
    bkd_model = copy.deepcopy(surrogate)

    topo_net = GraphTrojanNet(num_nodes)
    if args.feat_perb:
        feat_net = GraphTrojanNet(data.x.shape[1])
    else:
        feat_net = None

    bkd_edges = []
    for bi_step in range(args.bilevel_steps):
        topo_net, feat_net = train_gtn(
            args, bkd_model, topo_net, feat_net, topo_input, feat_input,
            topomask_train, featmask_train, trigger_data, adj, bkd_adj, device
        )
        topo_net = topo_net.to(device)
        rst_bkdA = topo_net(topo_input, topomask_train, args.topo_thrd, device, args.topo_activation, 'topo')
        _bkd_edges = torch.nonzero(rst_bkdA * (torch.ones_like(rst_bkdA) - adj))
        bkd_edges.extend(_bkd_edges.tolist())
        if only_addition:
            bkd_adj = torch.add(rst_bkdA * (torch.ones_like(rst_bkdA) - adj), adj)
        else:
            bkd_adj = torch.add(rst_bkdA, adj)   # only current position in cuda
        if args.feat_perb:
            rst_bkdX = feat_net(feat_input, featmask_train, args.feat_thrd, device, args.feat_activation, 'feat')
            bkd_features = torch.add(rst_bkdX, torch.tensor(trigger_data.features, device=device))

        bkd_data.edge_index = torch.cat(torch.where(bkd_adj > 0)).view(2, -1)
        if args.feat_perb:
            bkd_data.features = bkd_features.detach().cpu().numpy()
        bkd_model.continue_train(args, bkd_data, pos_set, neg_set, device)
        bkd_acc, test_acc = evaluate(bkd_model, bkd_data, trigger_nodes, device)
        # evaluate(bkd_model, trans_data, trigger_nodes, device)

        if abs(bkd_acc * 100 - 100) < 1e-3:
            print("Early Termination for 100% Attack Rate")
            break
    # print('Done')
    # args.transductive = False
    return bkd_model, topo_net, bkd_edges, target_label


def backdoor_attack_argument_group(group):
    group.add_argument('--trigger-size', type=int, default=3)
    group.add_argument('--num-triggers', type=int, default=20)
    group.add_argument('--bilevel-steps', type=int, default=4)
    group.add_argument('--gtn_lr', type=float, default=0.01)
    group.add_argument('--topo_thrd', type=float, default=0.5)
    group.add_argument('--gtn_epochs', type=int, default=10, help="# attack epochs")
    group.add_argument('--topo_activation', type=str, default='sigmoid', help="activation function for topology generator")
    group.add_argument('--gta_batch_size', type=int, default=16)
    group.add_argument('--lr_decay_steps', nargs='+', default=[25, 35], type=int)
    group.add_argument('--train_epochs', type=int, default=40)
    group.add_argument('--lambd', type=float, default=1)
    group.add_argument('--gtn_input_type', type=str, default='2hop')
    group.add_argument('--feat_thrd', type=float, default=0, help="threshold for feature generator (only useful for binary feature)")
    group.add_argument('--feat_activation', type=str, default='relu', help="activation function for feature generator")
    group.add_argument('--feat-perb', action='store_true')
    # group.add_argument('--transductive', action='store_true')


if __name__ == '__main__':
    parser = argument.load_parser()
    backdoor_group = parser.add_argument_group('Backdoor attack')
    backdoor_attack_argument_group(backdoor_group)

    args = parser.parse_args()
    # args.transductive = True

    data = data_loader.load(args)

    t0 = time.time()
    origin_model, bkd_model, topo_net = backdoor_attack(args, data)
    print('Total time of GTA,', int(time.time() - t0))


