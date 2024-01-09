import copy
import numpy as np
from scipy.special import comb
from decimal import Decimal
from tqdm import tqdm
from core_Ber import Smooth_Ber

import utils
import argument
import data_loader
from model.gnn import GNN
import nettack_adapter as ntk


def my_comb(d, m, _comb):
    if (d, m) not in _comb:
        _comb[(d, m)] = comb(d, m, exact=True)

    return _comb[(d, m)]

def my_powe(k, p, powe):
    if (k, p) not in powe:
        powe[(k, p)] = k ** p

    return powe[(k, p)]


def get_count(d, m, n, r, K, comb, powe):
    if r == 0 and m == 0 and n == 0:
        return 1
    # early stopping
    if (r == 0 and m != n) or min(m, n) < 0 or max(m, n) > d or m + n < r:
        return 0

    if r == 0:
        return my_comb(d, m, comb) * my_powe(K, m, powe)
    else:
        c = 0

        # the number which are assigned to the (d-r) dimensions
        for i in range(max(0, n-r), min(m, d-r, int(np.floor((m+n-r) * 0.5))) + 1):
            if (m+n-r) / 2 < i:
                break
            x = m - i
            y = n - i
            j = x + y - r
            # j = 0 ## if K = 1
            # the second one implies n <= m+r
            if j < 0 or x < j:
                continue
            tmp = my_powe(K-1, j) * my_comb(r, x-j) * my_comb(r-x+j, j)
            if tmp != 0:
                tmp *= my_comb(d-r, i) * my_powe(K, i)
                c += tmp

        return c


def count(num_nodes, k_range=[0,1], K=1):
    # global_comb = dict()
    # global_powe = dict()

    m_range = [0, num_nodes + 1]
    real_ttl = (K+1) ** num_nodes

    k2count = {}
    for k in range(k_range[0], k_range[1]):
        ttl = 0
        complete_cnt = []
        comb, powe = dict(), dict()
        for m in range(num_nodes + 1):
            # t0 = time.time()
            for n in range(m, min(m+k, num_nodes) + 1):
                c = get_count(num_nodes, m, n, k, K, comb, powe)
                if c != 0:
                    complete_cnt.append(((m, n), c))
                    ttl += c
                    if n > m:
                        ttl += c
            # if m % 100 == 0:
            #     print('r = {}, m = {:10d}/{:10d}, ttl ratio = {:.4f}, # of dict = {}'.format(k, m, m_range[1], ttl / real_ttl, len(complete_cnt)))
            #     print(args.dataset, len(powe), len(comb), int(time.time() - t0))

        k2count[k] = complete_cnt

    return k2count

def certify_K(p_a_bar, k2count, alpha, num_nodes, k=0):
    alpha = int(alpha * 100)
    beta = 100 - alpha
    
    # for k in range(k_range[0], k_range[1]):
    plower_z = int(p_a_bar * 100 ** 10) * (100 ** (num_nodes - 10))
    pupper_z = int((1-p_a_bar) * 100 ** 10) * (100 ** (num_nodes - 10))
    total_z = 100 ** num_nodes

    complete_cnt = k2count[k]
    raw_cnt = 0
    outcome = []
    for ((s, t), c) in complete_cnt:
        outcome.append((
        # likelihood ratio x flips s, x bar flips t
            # and then count, s, t
            (alpha ** (t - s)) * (beta ** (s - t)), c, s, t
        ))
        if s != t:    
            outcome.append((
                (alpha ** (s - t)) * (beta ** (t - s)), c, t, s
            ))

        raw_cnt += c
        if s != t:
            raw_cnt += c

    # sort likelihood ratio in a descending order, i.e., r1 >= r2 >= ...
    outcome_descend = sorted(outcome, key = lambda x: -x[0])
    p_given_lower = 0
    q_given_lower = 0
    for i in range(len(outcome_descend)):
        ratio, cnt, s, t = outcome_descend[i]
        p = (alpha ** (num_nodes - s)) * (beta ** s)
        q = (alpha ** (num_nodes - t)) * (beta ** t)
        q_delta_lower = q * cnt
        p_delta_lower = p * cnt

        if p_given_lower + p_delta_lower < plower_z:
            p_given_lower += p_delta_lower
            q_given_lower += q_delta_lower
        else:
            q_given_lower += (plower_z - p_given_lower) / Decimal(ratio)
            #q_given_lower += q * (plower_Z - p_given_lower) / Decimal(p)
            break
    q_given_lower /= total_z


    # sort likelihood ratio in a ascending order
    outcome_ascend = sorted(outcome, key = lambda x: x[0])
    p_given_upper = 0
    q_given_upper = 0
    for i in range(len(outcome_ascend)):
        ratio, cnt, s, t = outcome_ascend[i]
        p = (alpha ** (num_nodes - s)) * (beta ** s)
        q = (alpha ** (num_nodes - t)) * (beta ** t)
        q_delta_upper = q * cnt
        p_delta_upper = p * cnt

        if p_given_upper + p_delta_upper < pupper_z:
            p_given_upper += p_delta_upper
            q_given_upper += q_delta_upper
        else:
            q_given_upper += (pupper_z - p_given_upper) / Decimal(ratio)
            #q_given_upper += q * (pupper_Z - p_given_upper) / Decimal(p)
            break
    q_given_upper /= total_z

    # if q_given_lower - q_given_upper < 0:
    #     return True
    # print('q_given_lower', q_given_lower)
    # print('q_given_upper', q_given_upper)
    return q_given_lower < q_given_upper



def find_non_robust_nodes(args, surrogate, data, candidates):
    device = utils.get_device(args)

    k2count = count(data.num_nodes)

    adj = data.adjacency_matrix()
    smooth_classifier = Smooth_Ber(surrogate.model, data.num_classes, data.num_features, args.prob, adj, data.x, device)
    non_robust_nodes = []
    for i in tqdm(candidates, desc='certifing'):
        c_a, p_a_bar = smooth_classifier.certify_Ber(i, 100, 10000, args.c_alpha, args.c_batch)
        if not certify_K(p_a_bar, k2count, args.c_alpha, data.num_nodes):
            non_robust_nodes.append(i)
        # if len(non_robust_nodes) >= 2:
        #     break
    surrogate.model.cpu()
    return non_robust_nodes


# if __name__ == '__main__':
#     import torch
#     import pandas as pd
#     from collections import defaultdict
#     from torch_geometric.utils import to_undirected

#     parser = argument.load_parser()
#     parser.add_argument("--N0", type=int, default=100)
#     parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
#     parser.add_argument("--certify-alpha", dest='c_alpha', type=float, default=0.001, help="failure probability")
#     parser.add_argument("--certify-batch", dest='c_batch', type=int, default=10000, help="batch size")
#     parser.add_argument('--prob', default=0.8, type=float,
#                     help="probability to keep the status for each binary entry")

#     args = parser.parse_args()

#     device = utils.get_device(args)
#     data = data_loader.load(args)

#     bias = False

#     target = GNN(args, data.num_features, data.num_classes, surrogate=False, fix_seed=True, bias=bias)
#     target.train(data, device)
    
#     surrogate = GNN(args, data.num_features, data.num_classes, surrogate=False, fix_seed=False, bias=bias)
#     surrogate.train(data, device)

#     non_robust_nodes = find_non_robust_nodes(args, surrogate, data, data.test_set.nodes.tolist())
#     print(f'found {len(non_robust_nodes)} non-robust nodes as node tokens,')
#     print('  they are:', non_robust_nodes)
#     print('  Next, we are going to verify the fragility of these nodes.')
#     print('-' * 75)

#     result = defaultdict(list)
#     for v in non_robust_nodes:
#         clean_pred = target.predict(data, device, target_nodes=[v])
        
#         # Using nettack to check the robustness of the target node
#         nettack = ntk.adapte(surrogate, data, v, clean_pred, add_edge_only=True, epsilon=0.)
#         nettack.attack_surrogate(n_perturbations=3, perturb_features=False, perturb_structure=True)
#         perturbations = nettack.structure_perturbations

#         perturbation2pred = {}
#         verification_edges = []
#         for e in perturbations:
#             adv_data = copy.deepcopy(data)
#             adv_data.add_edges(torch.tensor([[e[0], e[1]], [e[1], e[0]]]))

#             adv = GNN(args, adv_data.num_features, adv_data.num_classes, surrogate=False, fix_seed=True, bias=bias)
#             adv.train(adv_data, device)

#             adv_pred = adv.predict(adv_data, device, target_nodes=[v])
#             perturbation2pred[e] = adv_pred[0]
#             if adv_pred[0] != clean_pred[0]:
#                 verification_edges.append(e)

#         if len(verification_edges) > 1:
#             adv_data = copy.deepcopy(data)
#             edge_index_t = to_undirected(torch.tensor(verification_edges).t()) 
#             adv_data.add_edges(edge_index_t)
#             adv = GNN(args, adv_data.num_features, adv_data.num_classes, surrogate=False, fix_seed=True, bias=bias)
#             adv.train(adv_data, device)
#             adv_pred = adv.predict(adv_data, device, target_nodes=[v])

#         result['target'].append(v)
#         result['label'].append(data.y[0])
#         result['clean prediction'].append(clean_pred[0])
#         result['perturbations'].append(perturbations)
#         result['single predictions'].append(list(perturbation2pred.values()))
#         result['union predictions'].append(adv_pred[0])

#     df = pd.DataFrame(result)
#     print('=' * 75)
#     print(' ' * 30 + 'Result')
#     print(df)
#     print('-' * 75)
#     df.to_csv('random_smoothing_result.csv')