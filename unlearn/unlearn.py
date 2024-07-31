""" An implementation of two unlearning methods: GraphEraser and CEU.

This module has three classes. A father class that contains all the common 
functions and two child classes that implement the two unlearning methods.

Typical usage example:
    1. Training
        unlearn_model = unlearn.GraphEraser(
            ft, adj, labels, 
            run_target.config[dataset][model.upper()], 
            device, model=model, verbose=verbose
        )
        res = unlearn_model.train()
        print('test accuracy: ', res['accuracy'])

    2. Unlearning
        res = unlearn_model.unlearn()
        print('unlearning test accuracy: ', res['accuracy'])
    
    3. Obtain posteriors of all nodes
        posteriors = unlearn_model.posterior()

Author:
    Kun Wu @ Stevens Institute of Technology

"""
import math
import copy
import random
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import pandas as pd
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from torch_geometric.utils import to_undirected
from scipy.optimize import fmin_ncg, fmin_cg


import utils
from model.gcn import GCN
from .hessian import hessian_vector_product


def to_vector(v):
    if isinstance(v, tuple) or isinstance(v, list):
        # return v.cpu().numpy().reshape(-1)
        return np.concatenate([vv.cpu().numpy().reshape(-1) for vv in v])
    else:
        return v.cpu().numpy().reshape(-1)


def to_list(v, sizes, device):
    _v = v
    result = []
    for size in sizes:
        total = reduce(lambda a, b: a * b, size)
        result.append(_v[:total].reshape(size).float().to(device))
        _v = _v[total:]
    return tuple(result)


def _mini_batch_hvp(x, **kwargs):
    model = kwargs['model']
    features = kwargs['features']
    x_train = kwargs['x_train']
    y_train = kwargs['y_train']
    adj = kwargs['adj']
    damping = kwargs['damping']
    device = kwargs['device']
    sizes = kwargs['sizes']
    p_idx = kwargs['p_idx']
    # use_torch = kwargs['use_torch']

    x = to_list(x, sizes, device)
    _hvp = hessian_vector_product(model, features, adj, x_train, y_train, x, device, p_idx)
    return [(a + damping * b).view(-1) for a, b in zip(_hvp, x)]


def _get_fmin_loss_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_loss(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        hvp = _mini_batch_hvp(x, **kwargs)
        obj = 0.5 * torch.dot(torch.cat(hvp, dim=0), x) - torch.dot(v, x)
        return obj.detach().cpu().numpy()

    return get_fmin_loss


def _get_fmin_grad_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_grad(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        hvp = _mini_batch_hvp(x, **kwargs)
        # return to_vector(hvp - v.view(-1))
        return (torch.cat(hvp, dim=0) - v).cpu().numpy()

    return get_fmin_grad


def _get_fmin_hvp_fn(v, **kwargs):
    device = kwargs['device']

    def get_fmin_hvp(x, p):
        p = torch.tensor(p, dtype=torch.float, device=device)
        hvp = _mini_batch_hvp(p, **kwargs)
        return to_vector(hvp)
    return get_fmin_hvp


def _get_cg_callback(v, **kwargs):
    device = kwargs['device']

    def cg_callback(x):
        x = torch.tensor(x, dtype=torch.float, device=device)
        hvp = _mini_batch_hvp(x, **kwargs)
        obj = 0.5 * torch.dot(torch.cat(hvp, dim=0), x) - torch.dot(v, x)
        # obj = 0.5 * torch.dot(hvp, x) - torch.dot(v.view(-1), x)
        # g = to_vector(hvp - v.view(-1))
        g = (torch.cat(hvp, dim=0) - v).cpu().numpy()
        print(f'loss: {obj:.4f}, grad: {np.linalg.norm(g):.8f}')
    return cg_callback


class Unlearn(object):
    
    def __init__(self, seed, features, adj, labels, config, device, model_type, epochs, verbose) -> None:
        self.seed = seed
        self.config = config
        self.device = device
        self.model_type = model_type
        self.epochs = epochs
        self.verbose = verbose

        self.idx_train = config['idx_train']
        self.idx_val = config['idx_val']
        self.idx_test = config['idx_test']
        # idx_random = np.arange(len(labels))
        # np.random.shuffle(idx_random)
        # self.idx_train = torch.tensor(idx_random[:int(len(labels) * config['train'])], device=self.device)
        # self.idx_val = torch.tensor(idx_random[int(len(labels) * config['train']):int(len(labels) * (config['train'] + config['val']))], device=self.device)
        # self.idx_test = torch.tensor(idx_random[int(len(labels) * (config['train'] + config['val'])):], device=self.device)

        # ss = StandardScaler()
        # self.features = torch.tensor(ss.fit_transform(features), dtype=torch.float, device=self.device)
        self.features = features.to(self.device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        self.labels = labels.to(self.device)
        self.adj = adj.to(self.device)
        self.adj_norm = utils.normalize(self.adj + torch.eye(self.adj.shape[0], device=self.device))
        self.num_nodes = self.adj.shape[0]

    def train(self):
        raise NotImplementedError("Please implement this function in child class")
    
    def posterior(self, indices=None):
        raise NotImplementedError("Please implement this function in child class")
    
    def unlearn(self, edges_to_forget):
        raise NotImplementedError("Please implement this function in child class")
    

class GraphEraser(Unlearn):

    def __init__(self, seed, features, adj, labels, 
                 config, device, model='gcn', epochs=1000, patience=10,
                 num_shards=20, partition='blpa', aggragation='lb', verbose=False):
        super().__init__(seed, features, adj, labels, config, device, model, epochs, verbose)
        self.num_shards = num_shards
        self.aggragation = aggragation
        self.patience = patience
        # self.aggragation = 'mean'


        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        if partition == 'blpa':
            self.shards = self._blpa()
        elif partition == 'bekm':
            pass
        else:
            raise ValueError("Please choose a valid partition method: 'blpa' or 'bekm'")

        if self.model_type.lower() == 'gcn':
            self.models = {shard: GCN(nfeat=self.features.shape[1], nhid=self.config['nhid'], 
                            nclass=int(self.labels.max().item() + 1), 
                            dropout=self.config['dropout']).to(self.device) for shard in range(self.num_shards)}
        elif self.model_type.lower() == 'gat':
            self.models = {shard: GAT(nfeat=self.features.shape[1], nhid=self.config['nhid'],
                               nclass=labels.max().item() + 1,
                               dropout=self.config['dropout'],
                               nhead=self.config['nheads']).to(self.device) for shard in range(self.num_shards)}
        else:
            print("graphsage later")
        
    def train(self): 
        for i in range(self.num_shards):
            shard = self.shards[i]
            model = self.models[i]
            model.reset_parameters()
            self._train_shard(shard, model, self.adj)

        if self.aggragation == 'lb': 
            _, metrics = self._optimal_aggr(self.idx_test)
        elif self.aggragation == 'mean':
            _, metrics = self._mean_aggr(self.idx_test)

        return metrics

    def unlearn(self, edges_to_forget): 
        # remove edges from the adjacency matrix
        self.adj_prime = copy.deepcopy(self.adj)

        _edge_index = to_undirected(torch.tensor(edges_to_forget).t())
        # _edge_index = to_undirected(torch.from_numpy(edges_to_forget.T))
        self.adj_prime[_edge_index[0], _edge_index[1]] = 0

        retrain_count = 0
        self.retrained_models = []
        for i in range(self.num_shards):
            shard = self.shards[i]
            model = copy.deepcopy(self.models[i])

            shard_nodes = torch.cat([torch.tensor(list(shard), device=self.device), torch.tensor(self.idx_val, device=self.device), torch.tensor(self.idx_test, device=self.device)])
            # check if the shard contains the edges to forget
            logit = np.isin(np.array(edges_to_forget).T, list(shard_nodes.cpu()))
            if (logit[0] & logit[1]).sum() == 0:
                self.retrained_models.append(model)    
                continue
            
            # retrain the shard if it contains the edges to forget
            self._train_shard(shard, model, self.adj_prime)
            retrain_count += 1
            self.retrained_models.append(model)
        
        print(f'unlearning done. {retrain_count} shards retrained.')
        _, metrics = self._optimal_aggr(self.idx_test, use_retrained=True)
        return metrics
    
    def predict(self, target_nodes, use_retrained=False):
        if self.aggragation == 'lb':
            posts, _ = self._optimal_aggr(target_nodes, use_retrained=use_retrained)
        elif self.aggragation == 'mean':
            posts, _ = self._mean_aggr(target_nodes)
        else:
            raise NotImplementedError('Please choose a valid aggragation method: "lb"')
        y_pred = torch.argmax(posts, dim=1).cpu().numpy()
        return y_pred

    def posterior(self, indices=None, use_retrained=False):
        if self.aggragation == 'lb':
            posts, _ = self._optimal_aggr(indices, use_retrained=use_retrained)
        elif self.aggragation == 'mean':
            posts, _ = self._mean_aggr(indices)
        else:
            raise NotImplementedError('Please choose a valid aggragation method: "lb"')

        return posts
    
    def _mean_aggr(self, indices=None, use_retrained=False):
        if indices is None:
            indices = torch.arange(self.num_nodes)
        _labels = self.labels[indices]

        posteriors = []
        for i in range(self.num_shards):
            shard = self.shards[i]
            if use_retrained:
                model = self.retrained_models[i]
            else:
                model = self.models[i]

            _shard_nodes = list(shard.union(set(indices)))
            # print('indices', indices)
            node2idx = {node: idx for idx, node in enumerate(_shard_nodes)}
            _indices = [node2idx[node] for node in indices]

            _features = self.features[_shard_nodes]
            _adj = self.adj[_shard_nodes, :][:, _shard_nodes]

            model.eval()
            with torch.no_grad():
                outputs = model(_features, _adj)[_indices]
                posteriors.append(outputs.unsqueeze(1))

        mean_posts = torch.mean(torch.cat(posteriors, dim=1), dim=1)

        y_preds = torch.argmax(mean_posts, dim=1)
        results = classification_report(_labels.cpu().numpy(), y_preds.cpu().numpy(), output_dict=True)
        return mean_posts, results


    def _optimal_aggr(self, indices=None, use_retrained=False):
        if indices is None:
            indices = torch.arange(self.num_nodes)
        _labels = self.labels[indices]

        alpha = self._train_alpha(use_retrained=use_retrained)

        posteriors = []
        for i in range(self.num_shards):
            shard = self.shards[i]
            if use_retrained:
                model = self.retrained_models[i]
            else:
                model = self.models[i]

            _shard_nodes = list(shard.union(set(indices)))
            node2idx = {node: idx for idx, node in enumerate(_shard_nodes)}
            _indices = [node2idx[node] for node in indices]

            _features = self.features[_shard_nodes]
            if use_retrained:
                _adj = self.adj_prime[_shard_nodes, :][:, _shard_nodes]
            else:
                _adj = self.adj[_shard_nodes, :][:, _shard_nodes]
            _adj_norm = utils.normalize(_adj + torch.eye(_adj.shape[0]).to(self.device))
            model.eval()
            with torch.no_grad():
                outputs = model(_features, _adj_norm)[_indices]
            posteriors.append(outputs)

        aggr_post = alpha[0] * posteriors[0]
        for s in range(1, self.num_shards):
            aggr_post += alpha[s] * posteriors[s]
        y_preds = torch.argmax(aggr_post, dim=1)

        result = classification_report(_labels.cpu().numpy(), y_preds.cpu().numpy(), output_dict=True)
        return aggr_post, result


    def _train_alpha(self, use_retrained=False):
        train_indices = np.random.choice(self.idx_train, size=1000, replace=False)
        train_indices = np.sort(train_indices).astype(int)
        _labels = self.labels[train_indices]

        _adj = self.adj_prime if use_retrained else self.adj
        _adj_norm = utils.normalize(_adj + torch.eye(_adj.shape[0]).to(self.device))
        
        posteriors = []
        for _, model in self.models.items():
            model.eval()
            with torch.no_grad():
                _posteriors = model(self.features, _adj_norm)[train_indices]
            posteriors.append(_posteriors.unsqueeze(0))
        posteriors = torch.cat(posteriors, dim=0)
        
        alpha = nn.Parameter(torch.full((self.num_shards, 1), fill_value=1 / self.num_shards, device=self.device), requires_grad=True)
        optimizer = torch.optim.Adam([alpha], lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        min_loss = math.inf
        for _ in range(100):
            optimizer.zero_grad()
            aggr_posteriors = torch.zeros_like(posteriors[0])
            for s in range(self.num_shards):
                aggr_posteriors += alpha[s] * posteriors[s]

            l1 = criterion(aggr_posteriors, _labels)
            l2 = torch.sqrt(torch.sum(torch.sum(torch.pow(alpha, 2))))
            loss = l1 + l2
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                alpha[:] = torch.clamp(alpha, min=0)

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_alpha = copy.deepcopy(alpha)

        return best_alpha / torch.sum(best_alpha)

    def _train_shard(self, shard, model, adj):
        _shard_nodes = torch.cat((torch.tensor(list(shard), device=self.device), torch.tensor(self.idx_val, device=self.device)))
        _idx_train = list(range(len(shard)))
        _idx_val = list(range(len(shard), len(shard) + len(self.idx_val)))

        _adj = adj[_shard_nodes, :][:, _shard_nodes]
        _adj_norm = utils.normalize(_adj + torch.eye(_adj.shape[0]).to(self.device))
        _features = self.features[_shard_nodes]
        _labels = self.labels[_shard_nodes]
        _labels_train = _labels[_idx_train]
        _labels_val = _labels[_idx_val]

        optimizer = torch.optim.Adam(model.parameters(),
                                lr=0.01,
                                weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        best_valid_loss = math.inf
        trial_count = 0
        best_model_state = model.state_dict()

        for e in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(_features, _adj_norm)[_idx_train]
            loss_train = criterion(output, _labels_train)
            loss_train.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                output = model(_features, _adj_norm)[_idx_val]
                loss_val = criterion(output, _labels_val)
                if torch.isnan(loss_val):
                    print(output.tolist())
                    print(_labels_val.tolist())
                    print(_shard_nodes.tolist())
                    raise ValueError('Loss is NaN.')

            if loss_val < best_valid_loss:
                best_valid_loss = loss_val
                trial_count = 0 
                best_model_state = model.state_dict()
            else:
                trial_count += 1
                if trial_count > self.patience:
                    break

            # print(f'Epoch {e+1:03d}: train loss {loss_train.item():.4f}, val loss {loss_val.item():.4f}')
        
        model.load_state_dict(best_model_state)
        return model

    def _blpa(self, iterations=30):
        node_threshold = math.ceil(len(self.idx_train) / self.num_shards)
        terminate_delta = 0

        # Initial shards
        np.random.shuffle(self.idx_train)
        shards = collections.defaultdict(set)
        node2shard = np.zeros(self.adj.shape[0])

        for shard, nodes in enumerate(np.array_split(self.idx_train, self.num_shards)):
            shards[shard] = set(nodes)
            node2shard[nodes] = shard

        _shards = copy.deepcopy(shards)

        # Run BLPA algorithm to partition the graph
        for _ in range(iterations):
            desire_move = self._blpa_determine_desire_move(node2shard)
            sort_indices = np.flip(np.argsort(desire_move[:, 2]))
            candidate_nodes = collections.defaultdict(list)

            for node in sort_indices:
                src_shard = desire_move[node][0]
                dst_shard = desire_move[node][1]

                if src_shard != dst_shard:
                    if len(shards[dst_shard]) < node_threshold:
                        node2shard[node] = dst_shard
                        shards[dst_shard].add(node)
                        if node in shards[src_shard]:
                            shards[src_shard].remove(node)

                        # reallocate the candidate nodes
                        candidate_nodes_cur = candidate_nodes[src_shard]
                        while len(candidate_nodes_cur) != 0:
                            node_cur = candidate_nodes_cur[0]
                            src_shard_cur = desire_move[node_cur][0]
                            dst_shard_cur = desire_move[node_cur][1]

                            node2shard[node_cur] = dst_shard_cur
                            shards[dst_shard_cur].add(node_cur)
                            shards[src_shard_cur].remove(node_cur)

                            candidate_nodes[dst_shard_cur].pop(0)
                            candidate_nodes_cur = candidate_nodes[src_shard_cur]
                    else:
                        candidate_nodes[dst_shard].append(node)
            delta = self._lpa_delta(_shards, shards)

            _shards = copy.deepcopy(shards)
            if delta <= terminate_delta:
                break

        return shards

    def _blpa_determine_desire_move(self, node2shard):
        desire_move = np.zeros([self.num_nodes, 3], dtype=np.int32)
        desire_move[:, 0] = node2shard

        for i in range(self.num_nodes):
            neighbor_community = node2shard[self.adj[i].long().cpu().numpy()]  # for bool adj
            unique_community, unique_count = np.unique(neighbor_community, return_counts=True)
            if unique_community.shape[0] == 0:
                continue
            max_indices = np.where(unique_count == np.max(unique_count))[0]

            if max_indices.size == 1:
                desire_move[i, 1] = unique_community[max_indices]
                desire_move[i, 2] = unique_count[max_indices]
            elif max_indices.size > 1:
                max_index = np.random.choice(max_indices)
                desire_move[i, 1] = unique_community[max_index]
                desire_move[i, 2] = unique_count[max_index]

        return desire_move
    
    def _lpa_delta(self, lpa_pre, lpa_cur):
        delta = 0.0
        for i in range(len(lpa_cur)):
            delta += len((lpa_cur[i] | lpa_pre[i]) - (lpa_cur[i] & lpa_pre[i]))

        return delta
    

class CEU(Unlearn):

    def __init__(self, seed, features, adj, labels, config, device, 
                 model_type='gcn', epochs=1000, patience=10, damping=0.01,
                 verbose=False) -> None:
        super().__init__(seed, features, adj, labels, config, device, model_type, epochs, verbose)
        self.patience = patience
        self.damping = 0.01

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(seed)
        random.seed(seed)
        if self.model_type.lower() == "gat":
            self.model = GAT(nfeat=self.features.shape[1],
                             nhid=self.config['nhid'],
                             nclass=labels.max().item() + 1,
                             dropout=self.config['dropout'],
                             nhead=self.config['nheads'])
        elif self.model_type.lower() == "gcn":
            self.model = GCN(nfeat=self.features.shape[1], 
                             nhid=self.config['nhid'], 
                             nclass=int(self.labels.max().item() + 1), 
                             dropout=self.config['dropout']).to(self.device)
        else:
            pass
            # model, enc1, enc2 = init_GraphSAGE(ft, adj, labels.max().item() + 1, config, device)
            # enc1.to(device)
            # enc2.to(device)


    def train(self):
        _labels_train = self.labels[self.idx_train]
        _labels_val = self.labels[self.idx_val]
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        best_valid_loss = math.inf
        trial_count = 0
        best_model_state = self.model.state_dict()

        for e in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(self.features, self.adj_norm)[self.idx_train]
            loss_train = criterion(output, _labels_train)
            loss_train.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                output = self.model(self.features, self.adj_norm)[self.idx_val]
                loss_val = criterion(output, _labels_val)

            if loss_val < best_valid_loss:
                best_valid_loss = loss_val
                trial_count = 0 
                best_model_state = self.model.state_dict()
            else:
                trial_count += 1
                if trial_count > self.patience:
                    break

            # print(f'Epoch {e+1:03d}: train loss {loss_train.item():.4f}, val loss {loss_val.item():.4f}')
        
        self.model.load_state_dict(best_model_state)

        # evaluate the learned model
        return self._evaluate(self.model, self.adj_norm)


    def unlearn(self, edges_to_forget):
        self.retrain_model = copy.deepcopy(self.model)
        self.adj_prime = copy.deepcopy(self.adj)

        # remove edges from the adjacency matrix
        _edge_index = to_undirected(torch.tensor(edges_to_forget).t())
        self.adj_prime[_edge_index[0], _edge_index[1]] = 0
        self.adj_prime_norm = utils.normalize(self.adj_prime + torch.eye(self.adj_prime.shape[0]).to(self.device))

        infected_nodes = self._infected_nodes(edges_to_forget, 2)
        infected_nodes = torch.tensor(infected_nodes, device=self.device)
        infected_labels = self.labels[infected_nodes]

        infl = self._influence(self.retrain_model, self.adj_prime_norm, infected_nodes, infected_labels)
        self._update_model_weight(self.retrain_model, infl)

        # Evaluate the performance of the retraine model
        # return self._evaluate(self.retrain_model, self.adj_prime)
    
    def predict(self, target_nodes, use_retrained=False):
        model = self.retrain_model if use_retrained else self.model
        adj_norm = self.adj_prime_norm if use_retrained else self.adj_norm

        model.eval()
        with torch.no_grad():
            outputs = model(self.features, adj_norm)[target_nodes]
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        return y_pred
    
    def posterior(self, indices=None, use_retrained=False):
        model = self.retrain_model if use_retrained else self.model
        adj_norm = self.adj_prime_norm if use_retrained else self.adj_norm

        model.eval()
        with torch.no_grad():
            outputs = model(self.features, adj_norm)

        return outputs

    def _evaluate(self, model, adj):
        model.eval()
        with torch.no_grad():
            outputs = model(self.features, adj)[self.idx_test]
        y_preds = outputs.argmax(dim=1)
        y_true = self.labels[self.idx_test].cpu().numpy()
        result = classification_report(y_true, y_preds.cpu().numpy(), output_dict=True)
        return result

    def _update_model_weight(self, model, infl):
        parameters = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            delta = [p + infl for p, infl in zip(parameters, infl)]
            for i, p in enumerate(parameters):
                p.copy_(delta[i])

    def _influence(self, model, adj_prime, infected_nodes, infected_labels):
        parameters = [p for p in model.parameters() if p.requires_grad]
        p = 1 / (len(self.idx_train))
        
        model.eval()
        output = model(self.features, adj_prime)[infected_nodes]
        loss1 = F.nll_loss(output, infected_labels)
        g1 = grad(loss1, parameters)

        output = model(self.features, self.adj_norm)[infected_nodes]
        loss2 = F.nll_loss(output, infected_labels)
        g2 = grad(loss2, parameters)

        v = [gg1 - gg2 for gg1, gg2 in zip(g1, g2)]

        ihvp, (cg_grad, status) = self.inverse_hvp_cg(model, v)
        I = [- p * i for i in ihvp]
        return I

    def inverse_hvp_cg(self, model, vs):
        inverse_hvp = []
        status, cg_grad = [], []

        parameters = [p for p in model.parameters() if p.requires_grad]
        for i, (v, p) in enumerate(zip(vs, parameters)):
            sizes = [p.size()]
            v = v.view(-1)

            fmin_loss_fn = _get_fmin_loss_fn(v, model=model,
                                             features = self.features,
                                             x_train=self.idx_train, y_train=self.labels[self.idx_train],
                                             adj=self.adj, damping=self.damping,
                                             sizes=sizes, p_idx=i, device=self.device)

            fmin_grad_fn = _get_fmin_grad_fn(v, model=model,
                                             features = self.features,
                                             x_train=self.idx_train, y_train=self.labels[self.idx_train],
                                             adj=self.adj, damping=self.damping,
                                             sizes=sizes, p_idx=i, device=self.device)
            '''fmin_hvp_fn = _get_fmin_hvp_fn(v, model=model,
                                           features = self.features,
                                           x_train=self.idx_train, y_train=self.labels[self.idx_train],
                                           adj=self.adj, damping=self.damping,
                                           sizes=sizes, p_idx=i, device=self.device)
            cg_callback = _get_cg_callback(v, model=model,
                                           features = self.features,
                                           x_train=self.idx_train, y_train=self.labels[self.idx_train],
                                           adj=self.adj, damping=self.damping,
                                           sizes=sizes, p_idx=i, device=self.device)'''
            res = fmin_cg(
                f=fmin_loss_fn,
                x0=to_vector(v),
                fprime=fmin_grad_fn,
                gtol=1E-4,
                # norm='fro',
                # callback=cg_callback,
                disp=False,
                full_output=True,
                maxiter=100,
            )
            #     res = fmin_ncg(
            #         f=fmin_loss_fn,
            #         x0=to_vector(v),
            #         fprime=fmin_grad_fn,
            #         fhess_p=fmin_hvp_fn,
            #         # callback=cg_callback,
            #         avextol=1e-5,
            #         disp=False,
            #         full_output=True,
            #         maxiter=100)

            inverse_hvp.append(to_list(torch.from_numpy(res[0]), sizes, self.device)[0])
            # inverse_hvp = to_list(torch.from_numpy(res[0]), sizes, device)
            # cg_grad = np.linalg.norm(fmin_grad_fn(res[0]))
            # status = res[4]
            # print('-----------------------------------')
            # cg_grad.append(np.linalg.norm(fmin_grad_fn(res[0]), ord=np.inf))

            # else:
            #     res = fmin_ncg(
            #         f=fmin_loss_fn,
            #         x0=to_vector(v),
            #         fprime=fmin_grad_fn,
            #         fhess_p=fmin_hvp_fn,
            #         # callback=cg_callback,
            #         avextol=1e-5,
            #         disp=False,
            #         full_output=True,
            #         maxiter=100)
            #     inverse_hvp.append(to_list(torch.from_numpy(res[0]), sizes, device)[0])
                # inverse_hvp = to_list(torch.from_numpy(res[0]), sizes, device)

            #     x, _err, d = fmin_l_bfgs_b(
            #         func=fmin_loss_fn,
            #         x0=to_vector(v),
            #         fprime=fmin_grad_fn,
            #         iprint=0,
            #     )
            #     inverse_hvp.append(to_list(x, sizes, device)[0])
            #     status.append(d['warnflag'])
            #     err += _err.item()
            # print('error:', err, status)
        return inverse_hvp, (cg_grad, status)


    def _infected_nodes(self, edges, l):
        assert l <= 2, 'Only support 1 or 2 hops GNNs'

        results = []
        if l == 1:
            for u, v in edges:
                results.extend(torch.where(self.adj[[u, v]] == 1)[1].tolist())
        elif l == 2:
            adj_2hop = self.adj.float() @ self.adj.float()
            for u, v in edges:
                results.extend(torch.where(adj_2hop[[u, v]] == 1)[1].tolist())
                results.extend([u, v])
        else:
            raise NotImplementedError('Only support 1 or 2 hops GNNs')

        return list(set(results))