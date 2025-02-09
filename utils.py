import math
import random
from collections import defaultdict
import numpy as np
from scipy.stats import entropy
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
import torch
from torch_geometric.utils import coalesce, to_undirected, is_undirected
from itertools import combinations
import scipy.sparse as sp

def get_device(args):
    if torch.backends.mps.is_built():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')
    return device

def sp_add_self_loops(adj, num_nodes=None):
    if num_nodes is None:
        num_nodes = adj.shape[0]
    i = torch.tensor([list(range(num_nodes)), list(range(num_nodes))])
    diag = torch.sparse_coo_tensor(i, torch.ones(num_nodes), (num_nodes, num_nodes))
    adj = adj + diag
    return adj

def normalize(mx):
    """
    sparse tensor version of gcn normalization
    """
    if mx.is_sparse:
        rowsum = mx.sum(dim=1).to_dense()
        r_inv = torch.pow(rowsum, -0.5).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.spmm(r_mat_inv, mx).mm(r_mat_inv.T)
    else:
        rowsum = mx.sum(dim=1)
        r_inv = torch.pow(rowsum, -0.5).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx).mm(r_mat_inv.T)
    return mx

def preprocess_adj(adj, ):
    N = adj.shape[0]
    if sp.isspmatrix(adj):
        adj_tilde = adj + sp.eye(N)
        degs_inv = np.power(adj_tilde.sum(0), -0.5)
        # adj_norm = adj
        adj_norm = adj_tilde.multiply(degs_inv).multiply(degs_inv.T)
    elif isinstance(adj, np.ndarray):
        adj_tilde = adj + np.eye(N)
        degs_inv = np.power(adj_tilde.sum(0), -0.5)
        adj_norm = np.multiply(np.multiply(adj_tilde, degs_inv[None,:]), degs_inv[:,None])

    return adj_norm

def confidence(logits):
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]
    uncertain_scores = sorted_logits[:, 0] - sorted_logits[:, 1]
    return uncertain_scores

def to_directed(edge_index):
    _edge_index = coalesce(edge_index)
    _edge_index = _edge_index[:, _edge_index[0] < _edge_index[1]]
    return _edge_index

def sample_edges(method, edge_index, num_removed_edges):
    assert is_undirected(edge_index), 'Invalid input, edge_index, it has to be undirected edges.'
    if method == 'random':
        num_edges = int(edge_index.size(1) / 2)
        _edge_index = coalesce(edge_index)
        _edge_index = _edge_index[:, _edge_index[0] < _edge_index[1]]
        random_indices = sorted(random.sample(range(num_edges), num_removed_edges))
        i = torch.zeros(num_edges).bool()
        i[random_indices] = 1
        removed_edges = _edge_index[:, i].t().tolist()
        remain_edge_index = to_undirected(_edge_index[:, ~i])
    else:
        raise NotImplementedError()

    return removed_edges, remain_edge_index

def on_boundary(z):
    values, counts = np.unique(z, return_counts=True)
    for c_idx in np.where(counts > 1)[0]:
        if c_idx == values.argmax():
            return True
    
    return False

def near_boundary(z, k):
    sorted_logits = np.sort(z)[::-1]
    return np.abs(sorted_logits[0] - sorted_logits[1]) <= k

def boundary_score(z):
    sorted_logits = np.sort(z)[::-1]
    return np.abs(sorted_logits[0] - sorted_logits[1])

def relu(x):
    return np.maximum(0, x)

def near_boundary_score(z, k):
    indices = np.arange(len(z))

    score_to_boundary = []
    for i, j in combinations(indices, 2):
        _z = z[np.where(~np.in1d(indices, [i, j]))] # remove i,j
        score = relu(z[i] - z[j] + k) + relu(_z.max() - z[i])
        score_to_boundary.append(score)

    return min(score_to_boundary)

def calc_q(df):
    attack_success_df = df[df['adv prediction'] != df['clean prediction']]
    if len(attack_success_df) > 0:
        q = np.sum(attack_success_df['# of flip'] > 0) / len(attack_success_df)
    else:
        raise ValueError('There is no successfully attacked node toknes')
    return q

def calc_q_tokens(df):
    node_tokens = pd.unique(df['target node'])
    node_token2q = {}
    num_unsuccessful_attack = 0
    for v in node_tokens:
        _df = df[df['target node'] == v]
        attack_success_df = _df[_df['adv prediction'] != _df['clean prediction']]
        if len(attack_success_df) > 0:
            q = np.sum(attack_success_df['# of flip'] > 0) / len(attack_success_df)
            node_token2q[v] = q
        else:
            node_token2q[v] = 0
            num_unsuccessful_attack += 1

    return node_token2q

def calc_q_by_trials(df, num_per_trial):
    qs = []
    for i in range(0, len(df), num_per_trial):
        _df = df[i: i+num_per_trial]
        attack_success_df = _df[_df['adv prediction'] != _df['clean prediction']]
        if len(attack_success_df) > 0:
            q = np.sum(attack_success_df['sub adv prediction'] == attack_success_df['clean prediction']) / len(attack_success_df)
        else:
            q = 1
        qs.append(q)
    return np.mean(qs)

def asr_against_num(df, numbers=[20, 40, 60, 80, 100]):
    for num in numbers:
        _df = df[:num]
        asr = (_df['clean prediction'] != _df['adv prediction']).sum() / num
        print(f'asr@{num}:', asr)

def calc_incorrectness_noise_by_trials(df, num_per_trial, sigma=0.1):
    result = []
    for i in range(0, len(df), num_per_trial):
        _df = df[i: i+num_per_trial]
        attack_success_df = _df[_df['adv prediction'] != _df['clean prediction']]
        if len(attack_success_df) > 0:
            # q = np.sum(attack_success_df['# of flip'] > 0) / len(attack_success_df)
            result.append(np.sum(attack_success_df[f'N@{sigma} prediction'] == attack_success_df['clean prediction']) / len(attack_success_df))
        else:
            result.append(1)

    return np.mean(result)

def calc_incorrectness_ii_by_trials(df, num_per_trial):
    result = []
    for i in range(0, len(df), num_per_trial):
        _df = df[i: i+num_per_trial]
        attack_success_df = _df[_df['adv prediction'] != _df['clean prediction']]
        if len(attack_success_df) > 0:
            # q = np.sum(attack_success_df['# of flip'] > 0) / len(attack_success_df)
            result.append(np.sum(attack_success_df['I prediction'] == attack_success_df['clean prediction']) / len(attack_success_df))
        else:
            result.append(1)
    return np.mean(result)

def calc_asr(df):
    num_success = np.sum(df['adv prediction'] != df['clean prediction'])
    asr = num_success / len(df)
    return asr

def calc_asr_by_tokens(df):
    node_tokens = pd.unique(df['target node'])

    node_token2asr = {}
    for v in node_tokens:
        _df = df[df['target node'] == v]
        num_success = np.sum(_df['adv prediction'] != _df['clean prediction'])
        node_token2asr[v] = num_success / len(_df)

    return node_token2asr

def calc_asr_by_trials(df, num_per_trial):
    ps = []
    for i in range(0, len(df), num_per_trial):
        _df = df[i: i+num_per_trial]
        num_success = np.sum(_df['adv prediction'] != _df['clean prediction'])
        ps.append(num_success / len(_df))
    return np.mean(ps)

def calc_removal_probability(df, q, num_trials):
    # asr = np.sum(df['adv prediction'] != df['clean prediction']) / num_node_tokens
    ps, ks = [], []
    for i in range(0, len(df), num_trials):
        _df = df[i: i+num_trials]
        num_node_tokens = len(_df)
        k = np.sum((_df['adv prediction'] != _df['clean prediction']) & (_df['# of flip'] == 0))
        ks.append(k)
        if k == 0:
            ps.append(0)
        else:
            p = math.comb(num_node_tokens, k) * (q ** k) * ((1 - q) ** (num_node_tokens - k))
            ps.append(p)
    print('k:', ks)
    return np.mean(ps)    

def calc_fpr(df, num_per_trials):
    fp_count_per_trial = []
    for i in range(0, len(df), num_per_trials):
        _df = df[i: i+num_per_trials]
        if 'adv prediction' in _df:
            fp_count_per_trial.append(np.sum(_df['adv prediction'] == _df['prediction']))
        else:
            fp_count_per_trial.append(np.sum(_df['bkd prediction'] == _df['prediction']))

    return np.mean(np.array(fp_count_per_trial) / num_per_trials)

def calc_fnr(df, num_per_trials):
    fn_count_per_trial = []
    for i in range(0, len(df), num_per_trials):
        _df = df[i: i+num_per_trials]
        fn_count_per_trial.append(_df['incomplete'].values.sum())
    return np.mean(np.array(fn_count_per_trial) / num_per_trials)


def homophily(nodes, edges, labels):
    L = defaultdict(int)
    degree = defaultdict(int)
    num_edges = len(edges)

    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
        if labels[u] == labels[v]:
            c = labels[u]
            L[c] += 1
      
    Q = 0
    for c in np.unique(labels):
        k_c = np.sum([degree[v] for v in np.where(labels == c)[0]])
        Q += L[c] /  num_edges - (k_c / (2 * num_edges)) ** 2
    return Q

    # num_nodes = len(nodes)
    # num_edges = torch.sum(adj) / 2

    # s = 0
    # for i in range(num_nodes):
    #     for j in range(i+1, num_nodes):
    #         i_deg = torch.sum(adj[i]).item()
    #         j_deg = torch.sum(adj[j]).item()
    #         s += adj[i, j].item() - (i_deg * j_deg) / (2 * num_edges.item()) if labels[i] == labels[j] else 0

    # return s / (2 * num_edges.item())

# def homophily_modularity(nodes, edges, labels):
#     # print(nodes, edges, labels)
#     G = nx.Graph()
#     G.add_nodes_from(list(range(len(nodes))))
#     for u,v in edges:
#         G.add_edge(u, v)
#     # print(G, G.edges, G.nodes, G.is_directed())
    

#     communities = []
#     for c in np.unique(labels):
#         communities.append(set(np.where(labels == c)[0].tolist()))

#     # print(communities)

#     return nx_comm.modularity(G, communities)


def homophily_entropy(num_classes, labels):
    label_dist = np.zeros((num_classes))
    for label, count in zip(*np.unique(labels, return_counts=True)):
        label_dist[label] = count
    label_dist /= np.sum(label_dist)
    return entropy(label_dist)

def kl_divergence(p, q):
    kl = np.where(p != 0, p * np.log(p / q), 0)
    kl = np.where(np.isnan(kl), 0, kl)
    kl = np.where(np.isinf(kl), 0, kl)
    return np.sum(kl)