import math
import copy
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import torch

from model.gnn import GNN
import utils


def normalize_add_perturb(adj, v, perturb):
    a = adj[v] + perturb
    inv_d = 1 + torch.sum(perturb)
    inv_d = 1./inv_d
    adj[v] = torch.mul(a, inv_d)
    return adj

def _add_perturb(adj, v, perturb, device=None):
    """
    # (1 - p) A + p(1 - A)

    """
    p = torch.zeros(adj.size(0), adj.size(0), device=device)
    p[v] = perturb
    p[:, v] = perturb

    # change the idx'th row and column
    p1 = torch.ones(adj.size(0), adj.size(0), device=device) - p
    adj2 = torch.ones(adj.size(0), adj.size(0), device=device) - torch.eye(adj.size(0), device=device) - adj

    perturbed_adj = torch.mul(p, adj2) + torch.mul(p1, adj)
    # perturbed_adj = torch.where(perturbed_adj > 0.5, torch.ones_like(perturbed_adj), torch.zeros_like(perturbed_adj))
    # print(torch.where(perturbed_adj == 1))
    # indices = torch.cat(torch.where(perturbed_adj == 1))
    # values = torch.ones(indices.size(1))
    # perturbed_adj = torch.sparse_coo_tensor(torch.cat(torch.where(perturbed_adj == 1)), values, size=(data.num_nodes, data.num_nodes))

    # _data = copy.deepcopy(data)
    # _data.update_edge_index_by_adj(perturbed_adj)
    # return _data
    return perturbed_adj

def add_perturb(input_adj, idx, perturb):
    # (1-x)A + x(1-A)
    x = np.zeros((input_adj.shape[0], input_adj.shape[1]))
    x[idx] = perturb  
    x[:,idx] = perturb

    
    #change the idx'th row and column
    x1 = np.ones((input_adj.shape[0], input_adj.shape[1])) - x
    adj2 = np.ones((input_adj.shape[0], input_adj.shape[1])) - input_adj

    for i in range(input_adj.shape[0]):   
        adj2[i][i] = 0

    perturbed_adj = np.multiply(x1, input_adj) + np.multiply(x, adj2)
    return perturbed_adj


def proj_lp(v, xi=8, p=2):
    # def proj_lp(v, xi=8, p=2):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    
    if p == 2:
        v = v * min(1, xi/torch.linalg.norm(v.flatten()))
    elif p == math.inf:
        v = torch.sign(v) * torch.minimum(abs(v), torch.tensor(xi))
    else:
        v = v
    #################
    v = torch.clip(v, 0, 1)
    ##################
    #to reduce the number of nonzero elements which means 
    #the times of perturbation, also prevents saddle point

    return v

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx, rowsum

def convert_to_v(adj, pert_m, degree, v):
    a = torch.mul(pert_m, degree)
    inv_m = torch.ones(adj.size(0)) - torch.mul(adj[v], 2)
    inv_m = torch.pow(inv_m, -1)
    res = torch.mul(a, inv_m)
    return res

def random_sample(model, data, ori_adj, v, p, prediction, device):
    K = 10000000
    print('statistic:', torch.mean(p), torch.max(p), torch.min(p))
    for i in range(K):
        sampled = np.random.binomial(1, p.cpu().detach())
        if sampled.sum() < 2:
            continue
        # print(np.sum(sampled == 1), np.sum(sampled == 2))
        s = torch.from_numpy(sampled).to(device)
        # print(sampled.sum())
        # if sampled.sum() > n_perturbations:
        #     continue
        # self.adj_changes.data.copy_(torch.tensor(sampled))
        # modified_adj = self.get_modified_adj(ori_adj)
        pert_adj = _add_perturb(ori_adj, v, s, device=device)
        pert_adj_norm = utils.normalize(pert_adj)
        pert_pred = model.predict(data, device, target_nodes=[v], pert_adj=pert_adj_norm)
        # loss = self._loss(output[idx_train], labels[idx_train])
        # loss = F.nll_loss(output[idx_train], labels[idx_train])
        # print(loss)
        # if best_loss < loss:
        #     best_loss = loss
        #     best_s = s
        if pert_pred[0] != prediction:
            break
    # self.adj_changes.data.copy_(torch.tensor(best_s))
    # print(torch.where(s == 1), torch.where(s == 2))
    # exit(0)
    return s

# def universal_attack(args, model: GNN, data, target_node):
#     device = utils.get_device(args)

#     fooling_rate = 0.
#     delta = 0.1
#     overshoot = 0.02
#     max_iter_df = 30
#     max_epoch = 100

#     v = torch.zeros(data.num_nodes)
#     cur_fooling_rate = 0.
#     epoch = 0

#     early_stop = 0

#     while fooling_rate < 1 - delta and epoch < max_epoch:
#         epoch += 1

#         train_nodes = data.train_set.nodes
#         np.random.shuffle(train_nodes)

#         for u in train_nodes:
#             data_prime = _add_perturb(u, v, device=device)
#             output, _, post = model.predict(data, device, train_nodes, return_posterior=True)

#             # if int(torch.argmax(post[u])) == int(torch.argmax())

def _iterative_minimum_perturbation(model, data, target_node, prediction, device, overshoot=0.02, max_iter=200, n_perturbation=2):
    v = torch.zeros(data.num_nodes, device=device)

    adj = torch.sparse_coo_tensor(
        data.edge_index, torch.ones(data.edge_index.size(1)),
        size=(data.num_nodes, data.num_nodes), device=device
    )
    _adj = copy.deepcopy(adj)
    _adj = utils.normalize(torch.eye(data.num_nodes, device=device) + _adj)

    for _ in range(1000):
        for _ in range(max_iter):
            v_delta = model.deepfool(target_node, data, _adj, prediction, device)
            v = v + v_delta
            v = torch.where(v < 0, torch.zeros_like(v), v)
            _adj = _add_perturb(_adj, target_node, (1 + overshoot) * v, device=device)
            _adj = torch.clip(_adj, 0, 1)

            pert_pred = model.predict(data, device, target_nodes=[target_node], pert_adj=_adj)
            if pert_pred[0] != prediction:
                # print('attack success!, prediction:', prediction, ', new:', pert_pred[0])
                break

        v = (1 + overshoot) * v
        _v = torch.where(v > 0.5, torch.ones_like(v), torch.zeros_like(v))
        if torch.sum(_v) < 2:
            continue

        v = _v
        pert_adj = _add_perturb(adj, target_node, v, device=device)
        l_prime = model.predict(data, device, target_nodes=[target_node], pert_adj=pert_adj)
        if l_prime[0] == prediction:
            break
    
    nodes = torch.where(v == 1)[0]
    perturbed_edges = [(target_node, u.item()) for u in nodes]
    # nodes = torch.argsort(v, descending=True)[:n_perturbation].tolist()
    # print('Are eddges exist?', [data.has_edge(target_node, u) for u in nodes])
    # perturbed_edges = [(target_node, u) for u in nodes]
        # v[:] = 0
        # v[nodes] = 1

        # pert_adj = _add_perturb(adj, target_node, v, device=device)
        # pert_adj = utils.normalize(torch.eye(data.num_nodes, data.num_nodes, device=device) + pert_adj)

        # pert_pred = model.predict(data, device, target_nodes=[target_node], pert_adj=pert_adj)
        # if pert_pred[0] != prediction:
        #     perturbed_edges = [(target_node, u) for u in nodes]
        #     break
        # else:
        #     print('Fail to flip,', i)

    # v = convert_to_v(adj, v, data.degree(target_node), target_node)
    # pert_adj = pert_data.adjacency_matrix().todense().numpy()
    # pert_adj = torch.where(_adj > 0.5, torch.ones_like(_adj), torch.zeros_like(_adj))
    # print('!!!', torch.sum((_adj - adj) > 0))
    # perturbed_edges = np.array(np.where(pert_adj.numpy() != adj.to_dense().numpy()))
    # v = proj_lp(v, p=math.inf) * 1000
    # v = proj_lp(v, p=math.inf) * 1000
    # v_norm = (v - v.min() ) / ( v.max() - v.min())
    # print('222', torch.mean(v), torch.max(v), torch.min(v))
    # pert_adj = torch.where(v >= 0.5, torch.ones_like(v), torch.zeros_like(v))
    # print('!!!', torch.sum(pert_adj), pert_adj.size())
    # exit(0)
    # perturbed_edges = np.array(np.where(pert_adj.numpy() != adj.to_dense()[target_node].numpy()))
    # perturbed_edges = perturbed_edges[:, perturbed_edges[0] < perturbed_edges[1]]
    return perturbed_edges

def iterative_minimum_perturbation(model, data, target_node, prediction, device, overshoot=0.02, max_iter=200, n_perturbation=2):
    v = torch.zeros(data.num_nodes, device=device)

    adj = torch.sparse_coo_tensor(
        data.edge_index, torch.ones(data.edge_index.size(1)),
        size=(data.num_nodes, data.num_nodes), device=device
    )
    _adj = copy.deepcopy(adj)
    # _adj2 = copy.deepcopy(adj)
    # _adj2 = utils.normalize(torch.eye(data.num_nodes, device=device) + _adj2)
    _adj = utils.normalize(torch.eye(data.num_nodes, device=device) + _adj)

    # for i in range(2, n_perturbation * 2 + 1):
    for _ in range(max_iter):
        v_delta = model.deepfool(target_node, data, _adj, prediction, device)
        v = v + v_delta
        v = torch.where(v < 0, torch.zeros_like(v), v)
        # print('&&&', torch.linalg.norm(v))

        # pert_adj = _g(adj, target_node, (1 + overshoot) * v)
        # pert_adj = utils.normalize(pert_adj)
        # pert_adj = torch.clip(pert_adj, 0, 1).to(device)

        _adj = _add_perturb(_adj, target_node, (1 + overshoot) * v, device=device)
        # _adj = normalize_add_perturb(_adj, target_node, (1 + overshoot) * v)
        _adj = torch.clip(_adj, 0, 1)

        pert_pred = model.predict(data, device, target_nodes=[target_node], pert_adj=_adj)
        if pert_pred[0] != prediction:
            print('attack success!, prediction:', prediction, ', new:', pert_pred[0])
            break

    v = (1 + overshoot) * v
    perb = random_sample(model, data, adj, target_node, v, prediction, device)
    nodes = torch.where(perb == 1)[0].tolist()
    # print('the number of perturbation:', torch.sum(perb == 1))
    # exit(0)
    # nodes = torch.argsort(v, descending=True)[:n_perturbation].tolist()
    print('Are eddges exist?', [data.has_edge(target_node, u) for u in nodes], perb.sum())
    perturbed_edges = [(target_node, u) for u in nodes]
        # v[:] = 0
        # v[nodes] = 1

        # pert_adj = _add_perturb(adj, target_node, v, device=device)
        # pert_adj = utils.normalize(torch.eye(data.num_nodes, data.num_nodes, device=device) + pert_adj)

        # pert_pred = model.predict(data, device, target_nodes=[target_node], pert_adj=pert_adj)
        # if pert_pred[0] != prediction:
        #     perturbed_edges = [(target_node, u) for u in nodes]
        #     break
        # else:
        #     print('Fail to flip,', i)

    # v = convert_to_v(adj, v, data.degree(target_node), target_node)
    # pert_adj = pert_data.adjacency_matrix().todense().numpy()
    # pert_adj = torch.where(_adj > 0.5, torch.ones_like(_adj), torch.zeros_like(_adj))
    # print('!!!', torch.sum((_adj - adj) > 0))
    # perturbed_edges = np.array(np.where(pert_adj.numpy() != adj.to_dense().numpy()))
    # v = proj_lp(v, p=math.inf) * 1000
    # v = proj_lp(v, p=math.inf) * 1000
    # v_norm = (v - v.min() ) / ( v.max() - v.min())
    # print('222', torch.mean(v), torch.max(v), torch.min(v))
    # pert_adj = torch.where(v >= 0.5, torch.ones_like(v), torch.zeros_like(v))
    # print('!!!', torch.sum(pert_adj), pert_adj.size())
    # exit(0)
    # perturbed_edges = np.array(np.where(pert_adj.numpy() != adj.to_dense()[target_node].numpy()))
    # perturbed_edges = perturbed_edges[:, perturbed_edges[0] < perturbed_edges[1]]
    return perturbed_edges


def minimum_attack(model, data, target_nodes, predictions, device, 
                   overshoot=1.02, max_epoch=400, max_iter=100, delta=0.9):
    p = torch.zeros(data.num_nodes, device=device)

    adj = data.adjacency_matrix().to(device)
    _adj = copy.deepcopy(adj)

    for _ in tqdm(range(max_epoch)):
        random_indices = np.arange(len(target_nodes))
        np.random.shuffle(random_indices) 
        for idx in random_indices:
            u, pred = target_nodes[idx], predictions[idx]
            _adj = _add_perturb(_adj, u, p, device=device)
            adj_norm = utils.normalize(torch.eye(data.num_nodes, device=device) + _adj)
            pert_pred, output = model.predict(data, device, target_nodes=[u], return_posterior=True, pert_adj=adj_norm)
            if pred == pert_pred[0]:
                v = torch.zeros(data.num_nodes, device=device)
                __adj = copy.deepcopy(_adj)
                _adj_norm = utils.normalize(torch.eye(data.num_nodes, device=device) + __adj)
                success_attack = False
                for i in range(max_iter):
                    delta_v = model.deepfool(u, data, _adj_norm, pred, device)
                    v = v + delta_v
                    v = torch.where(v < 0, torch.zeros_like(v), v)

                    pert_adj_norm = _add_perturb(_adj_norm, u, (1 + overshoot) * v, device).clip(0, 1)
                    _pert_pred = model.predict(data, device, target_nodes=[u], pert_adj=pert_adj_norm)
                    if _pert_pred[0] != pred:
                        success_attack = True
                        break
                if not success_attack:
                    print('Unsuccess attack') 
                else:
                    p = p + (1 + overshoot) * v
                    p = proj_lp(p, xi=4)

        print('norm:', torch.norm(p), torch.sum(p > 0.5))
        # if torch.sum(p > 0.5) == 0:
        #     continue
        p = torch.where(p>0.5, torch.ones_like(p), torch.zeros_like(p))
        if torch.sum(p) > 0:
            print('!!!', p.sum())

        pert_adj_norm = copy.deepcopy(adj)
        for u in target_nodes:
            pert_adj_norm = _add_perturb(pert_adj_norm, u, p, device=device)

        pert_predicrtions = model.predict(data, device, target_nodes=target_nodes, pert_adj=pert_adj_norm)

        err = np.sum(pert_predicrtions != predictions) / len(target_nodes)
        if err > delta:
            break

    nodes = torch.where(p == 1)[0]
    perturbations = [(u, v) for u in target_nodes for v in nodes]
    return perturbations