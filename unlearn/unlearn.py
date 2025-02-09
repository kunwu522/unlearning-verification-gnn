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
    Hide the author's name due to under reviewing.

"""
import os
import math
import copy
import json
import time
import random
from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import grad
import numpy as np
import pandas as pd
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, k_hop_subgraph, negative_sampling

import utils
from model.gcn import GCN
from model.sage import GraphSAGE
from .hessian import hessian_vector_product


class ConstrainedKmeans:
    def __init__(self, data_feat, num_clusters, node_threshold, terminate_delta, max_iteration=100):
        self.data_feat = data_feat
        self.num_clusters = num_clusters
        self.node_threshold = node_threshold
        self.terminate_delta = terminate_delta
        self.max_iteration = max_iteration

    def initialization(self):
        centroids = np.random.choice(np.arange(self.data_feat.shape[0]), self.num_clusters, replace=False)
        self.centroid = {}
        for i in range(self.num_clusters):
            self.centroid[i] = self.data_feat[centroids[i]]

    def clustering(self):
        centroid = copy.deepcopy(self.centroid)
        km_delta = []

        # pbar = tqdm(total=self.max_iteration)
        # pbar.set_description('Clustering')

        for i in range(self.max_iteration):
            # print('iteration %s' % (i,))

            self._node_reassignment()
            self._centroid_updating()

            # record the average change of centroids, if the change is smaller than a very small value, then terminate
            delta = self._centroid_delta(centroid, self.centroid)
            km_delta.append(delta)
            centroid = copy.deepcopy(self.centroid)

            if delta <= self.terminate_delta:
                break
            # print("delta: %s" % delta)
        # pbar.close()
        return self.clusters, km_delta

    def _node_reassignment(self):
        self.clusters = {}
        for i in range(self.num_clusters):
            self.clusters[i] = np.zeros(0, dtype=np.uint64)

        distance = np.zeros([self.num_clusters, self.data_feat.shape[0]])

        for i in range(self.num_clusters):
            distance[i] = np.sum(np.power((self.data_feat - self.centroid[i]), 2), axis=1)

        sort_indices = np.unravel_index(np.argsort(distance, axis=None), distance.shape)
        clusters = sort_indices[0]
        users = sort_indices[1]
        selected_nodes = np.zeros(0, dtype=np.int64)
        counter = 0

        while len(selected_nodes) < self.data_feat.shape[0]:
            cluster = int(clusters[counter])
            user = users[counter]
            if self.clusters[cluster].size < self.node_threshold:
                self.clusters[cluster] = np.append(self.clusters[cluster], np.array(int(user)))
                selected_nodes = np.append(selected_nodes, np.array(int(user)))

                # delete all the following pairs for the selected user
                user_indices = np.where(users == user)[0]
                a = np.arange(users.size)
                b = user_indices[user_indices > counter]
                remain_indices = a[np.where(np.logical_not(np.isin(a, b)))[0]]
                clusters = clusters[remain_indices]
                users = users[remain_indices]

            counter += 1

    def _centroid_updating(self):
        for i in range(self.num_clusters):
            self.centroid[i] = np.mean(self.data_feat[self.clusters[i].astype(int)], axis=0)

    def _centroid_delta(self, centroid_pre, centroid_cur):
        delta = 0.0
        for i in range(len(centroid_cur)):
            delta += np.sum(np.abs(centroid_cur[i] - centroid_pre[i]))

        return delta


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

        self.edge_index = torch.cat(torch.where(self.adj > 0)).view(2, -1).to(self.device)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def train(self):
        raise NotImplementedError("Please implement this function in child class")
    
    def posterior(self, indices=None):
        raise NotImplementedError("Please implement this function in child class")
    
    def unlearn(self, edges_to_forget, return_num_retrain=False):
        raise NotImplementedError("Please implement this function in child class")

    def predict(self, target_nodes, use_retrained=False, return_posterior=False):
        raise NotImplementedError("Please implement this function in child class")
    
    def _evaluate(self, model, adj):
        model.eval()
        with torch.no_grad():
            outputs = model(self.features, adj)[self.idx_test]
        y_preds = outputs.argmax(dim=1)
        y_true = self.labels[self.idx_test].cpu().numpy()
        result = classification_report(y_true, y_preds.cpu().numpy(), output_dict=True, zero_division=0)
        return result

class GraphEraser(Unlearn):

    def __init__(self, seed, features, adj, labels, 
                 config, device, model_type='gcn', epochs=1000, patience=10,
                 num_shards=20, partition='blpa', aggregation='lb', verbose=False):
        super().__init__(seed, features, adj, labels, config, device, model_type, epochs, verbose)
        self.num_shards = num_shards
        self.aggregation = aggregation
        self.patience = patience
        # self.aggregation = 'mean'

        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # np.random.seed(seed)
        # random.seed(seed)

        if partition == 'blpa':
            self.shards = self._blpa()
        elif partition == 'bekm':
            self.shards = self._bekm()
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

        if self.aggregation == 'lb': 
            _, metrics = self._optimal_aggr(self.idx_test)
        elif self.aggregation == 'mean':
            _, metrics = self._mean_aggr(self.idx_test)

        return metrics

    def unlearn(self, edges_to_forget, return_num_retrain=False): 
        # remove edges from the adjacency matrix
        self.adj_prime = copy.deepcopy(self.adj)

        _edge_index = to_undirected(torch.tensor(edges_to_forget).t())
        # _edge_index = to_undirected(torch.from_numpy(edges_to_forget.T))
        # print('edge to be removed in adj:', self.adj_prime[_edge_index[0], _edge_index[1]])
        self.adj_prime[_edge_index[0], _edge_index[1]] = 0
        # print('edge to be removed in adj after:', self.adj_prime[_edge_index[0], _edge_index[1]])

        retrain_count = 0
        self.retrained_models = {}
        for i in range(self.num_shards):
            shard = self.shards[i]
            model = copy.deepcopy(self.models[i])

            shard_nodes = torch.cat([
                torch.tensor(list(shard), device=self.device), 
                torch.tensor(self.idx_val, device=self.device), 
                torch.tensor(self.idx_test, device=self.device)
            ])
            # check if the shard contains the edges to forget
            logit = np.isin(np.array(edges_to_forget).T, list(shard_nodes.cpu()))
            if (logit[0] & logit[1]).sum() == 0:
                self.retrained_models[i] = model
                continue
            
            # retrain the shard if it contains the edges to forget
            model.reset_parameters()
            self._train_shard(shard, model, self.adj_prime)
            retrain_count += 1
            self.retrained_models[i] = model
        
        # print(f'unlearning done. {retrain_count} shards retrained.')
        _, metrics = self._optimal_aggr(self.idx_test, use_retrained=True)
        if return_num_retrain:
            return metrics, retrain_count
        else:
            return metrics
    
    def predict(self, target_nodes, use_retrained=False, return_posterior=False):
        if self.aggregation == 'lb':
            posts, _ = self._optimal_aggr(target_nodes, use_retrained=use_retrained)
        elif self.aggregation == 'mean':
            posts, _ = self._mean_aggr(target_nodes)
        else:
            raise NotImplementedError('Please choose a valid aggregation method: "lb"')
        y_pred = torch.argmax(posts, dim=1).cpu().numpy()
        if return_posterior:
            return y_pred, posts.detach().cpu()
        else:
            return y_pred

    def posterior(self, indices=None, use_retrained=False):
        if self.aggregation == 'lb':
            posts, _ = self._optimal_aggr(indices, use_retrained=use_retrained)
        elif self.aggregation == 'mean':
            posts, _ = self._mean_aggr(indices)
        else:
            raise NotImplementedError('Please choose a valid aggregation method: "lb"')

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
        results = classification_report(_labels.cpu().numpy(), y_preds.cpu().numpy(), output_dict=True, zero_division=0)
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
            _shard_nodes = list(shard.union(set(indices + self.idx_val + self.idx_test)))
            node2idx = {node: idx for idx, node in enumerate(_shard_nodes)}
            _indices = [node2idx[node] for node in indices]

            _features = self.features[_shard_nodes]
            _labels = self.labels[_shard_nodes][_indices]
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
        loss = F.cross_entropy(aggr_post, self.labels[indices])
        for s in range(1, self.num_shards):
            aggr_post += alpha[s] * posteriors[s]
        y_preds = torch.argmax(aggr_post, dim=1)

        result = classification_report(_labels.cpu().numpy(), y_preds.cpu().numpy(), output_dict=True, zero_division=0)
        result['loss'] = loss
        return aggr_post, result


    def _train_alpha(self, use_retrained=False):
        train_indices = np.random.choice(self.idx_train, size=1000, replace=False)
        train_indices = np.sort(train_indices).astype(int)
        _labels = self.labels[train_indices]

        _adj = self.adj_prime if use_retrained else self.adj
        _adj_norm = utils.normalize(_adj + torch.eye(_adj.shape[0]).to(self.device))
        
        posteriors = []
        models = self.retrained_models if use_retrained else self.models
        for _, model in models.items():
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
        _shard_nodes = torch.cat((
            torch.tensor(list(shard), device=self.device), 
            torch.tensor(self.idx_val, device=self.device),
            torch.tensor(self.idx_test, device=self.device)
        ))
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

        # for e in range(self.epochs):
        for e in range(100):
            model.train()
            optimizer.zero_grad()
            output = model(_features, _adj_norm)[_idx_train]
            loss_train = criterion(output, _labels_train)
            loss_train.backward()
            optimizer.step()

            # model.eval()
            # with torch.no_grad():
            #     output = model(_features, _adj_norm)[_idx_val]
            #     loss_val = criterion(output, _labels_val)
            #     if torch.isnan(loss_val):
            #         print(output.tolist())
            #         print(_labels_val.tolist())
            #         print(_shard_nodes.tolist())
            #         raise ValueError('Loss is NaN.')

            # if loss_val < best_valid_loss:
            #     best_valid_loss = loss_val
            #     trial_count = 0 
            #     best_model_state = model.state_dict()
            # else:
            #     trial_count += 1
            #     if trial_count > self.patience:
            #         break

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
    
    def _node_embeddings(self):
        embedding_size = self.features.shape[1]
        _model = GraphSAGE(
            nfeat=self.features.shape[1], 
            nhid=embedding_size, 
            nclass=int(self.labels.max().item() + 1), 
            dropout=self.config['dropout']
        ).to(self.device)

        optimizer = torch.optim.Adam(_model.parameters(), lr=self.config['lr'], weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(100):
            _model.train()
            optimizer.zero_grad()
            output = _model(self.features, self.edge_index)[self.idx_train]
            loss_train = criterion(output, self.labels[self.idx_train])
            loss_train.backward()
            optimizer.step()

        _model.eval()
        with torch.no_grad():
            embeddings = _model(self.features, self.edge_index)
        return embeddings.cpu().numpy()

    def _bekm(self):
        train_idx = self.idx_train
        idx_mapping = {idx: node for idx, node in enumerate(train_idx)}
        node_threshold = math.ceil(len(self.features) / self.num_shards)

        embeddings = self._node_embeddings()[train_idx]
        cluster = ConstrainedKmeans(embeddings, self.num_shards, node_threshold, 0)
        cluster.initialization()
        shards, _ = cluster.clustering()
        result = {}
        for i in range(self.num_shards):
            result[i] = set([idx_mapping[int(s)] for s in shards[i]])
        return result

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
    

class DeletionLayer(nn.Module):
    """
    A custom layer that applies deletion operator to the local
    https://github.com/mims-harvard/GNNDelete

    """
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        # self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 100)
        self.deletion_weight = nn.Parameter(torch.zeros(dim, dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.deletion_weight)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x


class GCNDelete(GCN):
    def __init__(self, nfeat, nhid, nclass, droput):
        super(GCNDelete, self).__init__(nfeat, nhid, nclass, droput)
    # def __init__(self, in_dim, hidden_dim, out_dim, **kwargs):
    #     super(GCN_delete, self).__init__()

    #     self.conv1 = GCNConv(in_dim, hidden_dim)
    #     self.conv2 = GCNConv(hidden_dim, out_dim)
    #     # self.dropout = nn.Dropout(args.dropout)

    # def forward(self, x, edge_index, return_all_emb=False):
    #     x1 = self.conv1(x, edge_index)
    #     x = F.relu(x1)
    #     # x = self.dropout(x)
    #     x2 = self.conv2(x, edge_index)

    #     if return_all_emb:
    #         return x1, x2
        
    #     return x2

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1).long()
            # logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
            logits = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)

        else:
            edge_index = pos_edge_index.long()
            # logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
            logits = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)

        return logits

class GCNDeletion(GCNDelete):
    # def __init__(self, in_dim, hidden_dim, out_dim, mask_1hop=None, mask_2hop=None):
    def __init__(self, nfeat, nhid, nclass, dropout, mask_1hop=None, mask_2hop=None):
        super(GCNDeletion, self).__init__(nfeat, nhid, nclass, dropout)
        self.deletion1 = DeletionLayer(nhid, mask_1hop)
        self.deletion2 = DeletionLayer(nclass, mask_2hop)

        # self.gc1.weight.requires_grad = False
        # self.gc2.bias.requires_grad = False
        for p in self.gc1.parameters():
            p.requires_grad_(False)
        for p in self.gc2.parameters():
            p.requires_grad_(False)

    def forward(self, x, adj, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        # with torch.no_grad():
        x1 = self.gc1(x, adj)
        
        x1 = self.deletion1(x1, mask_1hop)

        x = F.relu(x1)
        
        x2 = self.gc2(x, adj)
        x2 = self.deletion2(x2, mask_2hop)

        if return_all_emb:
            return x1, x2
        return x2
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)
    
@torch.no_grad()
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

class Trainer:
    def __init__(self, args, device='cpu'):
        self.args = args
        self.device = device
        self.trainer_log = {
            'unlearning_model': args.unlearning_model, 
            'dataset': args.dataset, 
            'log': []}
        self.logit_all_pair = None
        self.df_pos_edge = []

        with open(os.path.join(self.args.checkpoint_dir, 'training_args.json'), 'w') as f:
            json.dump(vars(args), f)

    def freeze_unused_weights(self, model, mask):
        grad_mask = torch.zeros_like(mask)
        grad_mask[mask] = 1

        model.deletion1.deletion_weight.register_hook(lambda grad: grad.mul_(grad_mask))
        model.deletion2.deletion_weight.register_hook(lambda grad: grad.mul_(grad_mask))
    
    @torch.no_grad()
    def get_link_labels(self, pos_edge_index, neg_edge_index):
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    @torch.no_grad()
    def get_embedding(self, model, data, on_cpu=False):
        original_device = next(model.parameters()).device

        if on_cpu:
            model = model.cpu()
            data = data.cpu()
        
        z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])

        model = model.to(original_device)

        return z

    def train(self, model, data, optimizer, args):
        # if self.args.dataset in ['Cora', 'PubMed', 'DBLP', 'CS']:
        return self.train_fullbatch(model, data, optimizer, args)

        # if self.args.dataset in ['Physics']:
        #     return self.train_minibatch(model, data, optimizer, args)

        # if 'ogbl' in self.args.dataset:
        #     return self.train_minibatch(model, data, optimizer, args)

    def train_fullbatch(self, model, data, optimizer, args):
        start_time = time.time()
        best_valid_loss = 1000000

        data = data.to(self.device)
        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            # Positive and negative sample
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.dtrain_mask.sum())
            
            z = model(data.x, data.train_pos_edge_index)
            # edge = torch.cat([train_pos_edge_index, neg_edge_index], dim=-1)
            # logits = model.decode(z, edge[0], edge[1])
            logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
            label = get_link_labels(data.train_pos_edge_index, neg_edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, label)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item()
                } 
                for log in [train_log, valid_log]:
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))

        self.trainer_log['training_time'] = time.time() - start_time

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid loss = {best_valid_loss:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_valid_loss'] = best_valid_loss

    def train_minibatch(self, model, data, optimizer, args):
        start_time = time.time()
        best_valid_loss = 1000000

        data.edge_index = data.train_pos_edge_index
        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=args.batch_size, walk_length=2, num_steps=args.num_steps,
        )
        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            epoch_loss = 0
            for step, batch in enumerate(tqdm(loader, desc='Step', leave=False)):
                # Positive and negative sample
                train_pos_edge_index = batch.edge_index.to(device)
                z = model(batch.x.to(device), train_pos_edge_index)

                neg_edge_index = negative_sampling(
                    edge_index=train_pos_edge_index,
                    num_nodes=z.size(0))
                
                logits = model.decode(z, train_pos_edge_index, neg_edge_index)
                label = get_link_labels(train_pos_edge_index, neg_edge_index)
                loss = F.binary_cross_entropy_with_logits(logits, label)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                log = {
                    'epoch': epoch,
                    'step': step,
                    'train_loss': loss.item(),
                }
                wandb.log(log)
                msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                tqdm.write(' | '.join(msg))

                epoch_loss += loss.item()

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss / step
                }
                
                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))

        self.trainer_log['training_time'] = time.time() - start_time

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid loss = {best_valid_loss:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_valid_loss'] = best_valid_loss
        self.trainer_log['training_time'] = np.mean([i['epoch_time'] for i in self.trainer_log['log'] if 'epoch_time' in i])

    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False, device='cuda'):    
        model.eval()
        pos_idx = random.sample(range(data.edge_index.size(1)), 200)
        pos_edge_index = data.edge_index[:, pos_idx].to(device)
        neg_edge_index = negative_sampling(pos_edge_index, data.num_nodes, num_neg_samples=pos_edge_index.size(1)).to(device)
        mask = data.dr_mask

        _adj = torch.zeros(data.num_nodes, data.num_nodes)
        _adj[data.train_pos_edge_index[0, mask], data.train_pos_edge_index[1, mask]] = 1
        _adj[data.train_pos_edge_index[1, mask], data.train_pos_edge_index[0, mask]] = 1
        _adj_norm = utils.normalize(_adj + torch.eye(data.num_nodes)).to(device)
        z = model(data.x.to(device), _adj_norm)
        # logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        _edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1).long()
        logits = (z[_edge_index[0]] * z[_edge_index[1]]).sum(dim=-1).sigmoid()
        label = self.get_link_labels(pos_edge_index, neg_edge_index)

        # DT AUC AUP
        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        val_auc = roc_auc_score(label.cpu(), logits.cpu())
        # dt_aup = average_precision_score(label.cpu(), logits.cpu())

        # DF AUC AUP
        # if self.args.unlearning_model in ['original']:
        #     df_logit = []
        # else:
        #     # df_logit = model.decode(z, data.train_pos_edge_index[:, data.df_mask]).sigmoid().tolist()
        #     df_logit = model.decode(z, data.directed_df_edge_index).sigmoid().tolist()

        # if len(df_logit) > 0:
        #     df_auc = []
        #     df_aup = []
        
        #     # Sample pos samples
        #     if len(self.df_pos_edge) == 0:
        #         for i in range(500):
        #             mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
        #             idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
        #             mask[idx] = True
        #             self.df_pos_edge.append(mask)
            
        #     # Use cached pos samples
        #     for mask in self.df_pos_edge:
        #         pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()
                
        #         logit = df_logit + pos_logit
        #         label = [0] * len(df_logit) +  [1] * len(df_logit)
        #         df_auc.append(roc_auc_score(label, logit))
        #         df_aup.append(average_precision_score(label, logit))
        
        #     df_auc = np.mean(df_auc)
        #     df_aup = np.mean(df_aup)

        # else:
        #     df_auc = np.nan
        #     df_aup = np.nan

        # Logits for all node pairs
        # if pred_all:
        #     logit_all_pair = (z @ z.t()).cpu()
        # else:
        #     logit_all_pair = None

        # log = {
        #     f'{stage}_loss': loss,
        #     f'{stage}_dt_auc': dt_auc,
        #     f'{stage}_dt_aup': dt_aup,
        #     f'{stage}_df_auc': df_auc,
        #     f'{stage}_df_aup': df_aup,
        #     f'{stage}_df_logit_mean': np.mean(df_logit) if len(df_logit) > 0 else np.nan,
        #     f'{stage}_df_logit_std': np.std(df_logit) if len(df_logit) > 0 else np.nan
        # }

        # if self.args.eval_on_cpu:
        #     model = model.to(device)

        # return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, loga
        return loss, val_auc

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        
        if ckpt == 'best':    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))
            model.load_state_dict(ckpt['model_state'])

        if 'ogbl' in self.args.dataset:
            pred_all = False
        else:
            pred_all = True
        loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log = self.eval(model, data, 'test', pred_all)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_auc'] = dt_auc
        self.trainer_log['dt_aup'] = dt_aup
        self.trainer_log['df_logit'] = df_logit
        self.logit_all_pair = logit_all_pair
        self.trainer_log['df_auc'] = df_auc
        self.trainer_log['df_aup'] = df_aup
        self.trainer_log['auc_sum'] = dt_auc + df_auc
        self.trainer_log['aup_sum'] = dt_aup + df_aup
        self.trainer_log['auc_gap'] = abs(dt_auc - df_auc)
        self.trainer_log['aup_gap'] = abs(dt_aup - df_aup)

        # # AUC AUP on Df
        # if len(df_logit) > 0:
        #     auc = []
        #     aup = []

        #     if self.args.eval_on_cpu:
        #         model = model.to('cpu')
            
        #     z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])
        #     for i in range(500):
        #         mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
        #         idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
        #         mask[idx] = True
        #         pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()

        #         logit = df_logit + pos_logit
        #         label = [0] * len(df_logit) +  [1] * len(df_logit)
        #         auc.append(roc_auc_score(label, logit))
        #         aup.append(average_precision_score(label, logit))

        #     self.trainer_log['df_auc'] = np.mean(auc)
        #     self.trainer_log['df_aup'] = np.mean(aup)


        if model_retrain is not None:    # Deletion
            self.trainer_log['ve'] = verification_error(model, model_retrain).cpu().item()
            # self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, data=data).cpu().item()

        # MI Attack after unlearning
        if attack_model_all is not None:
            mi_logit_all_after, mi_sucrate_all_after = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_after'] = mi_logit_all_after
            self.trainer_log['mi_sucrate_all_after'] = mi_sucrate_all_after
        if attack_model_sub is not None:
            mi_logit_sub_after, mi_sucrate_sub_after = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_after'] = mi_logit_sub_after
            self.trainer_log['mi_sucrate_sub_after'] = mi_sucrate_sub_after
            
            self.trainer_log['mi_ratio_all'] = np.mean([i[1] / j[1] for i, j in zip(self.trainer_log['mi_logit_all_after'], self.trainer_log['mi_logit_all_before'])])
            self.trainer_log['mi_ratio_sub'] = np.mean([i[1] / j[1] for i, j in zip(self.trainer_log['mi_logit_sub_after'], self.trainer_log['mi_logit_sub_before'])])
            print(self.trainer_log['mi_ratio_all'], self.trainer_log['mi_ratio_sub'], self.trainer_log['mi_sucrate_all_after'], self.trainer_log['mi_sucrate_sub_after'])
            print(self.trainer_log['df_auc'], self.trainer_log['df_aup'])

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log

    @torch.no_grad()
    def get_output(self, model, node_embedding, data):
        model.eval()
        node_embedding = node_embedding.to(device)
        edge = data.edge_index.to(device)
        output = model.decode(node_embedding, edge, edge_type)

        return output

    def save_log(self):
        # print(self.trainer_log)
        with open(os.path.join(self.args.checkpoint_dir, 'trainer_log.json'), 'w') as f:
            json.dump(self.trainer_log, f)
        
        torch.save(self.logit_all_pair, os.path.join(self.args.checkpoint_dir, 'pred_proba.pt'))


@torch.no_grad()
def member_infer_attack(target_model, attack_model, data, logits=None, device='cpu'):
    '''Membership inference attack'''

    edge = data.train_pos_edge_index[:, data.df_mask]
    z = target_model(data.x.to(device), data.train_pos_edge_index[:, data.dr_mask])
    feature1 = target_model.decode(z, edge).sigmoid()
    feature0 = 1 - feature1
    feature = torch.stack([feature0, feature1], dim=1)
    # feature = torch.cat([z[edge[0]], z[edge][1]], dim=-1)
    logits = attack_model(feature)
    _, pred = torch.max(logits, 1)
    suc_rate = 1 - pred.float().mean()

    return torch.softmax(logits, dim=-1).squeeze().tolist(), suc_rate.cpu().item()
    
class GNNDeleteTrainer(Trainer):

    def train(self, model, data, retrain_data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        return self.train_fullbatch(model, data, retrain_data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)
        # if 'ogbl' in self.args.dataset:
        #     return self.train_minibatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

        # else:
        #     return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    def compute_loss(self, model, data, random_loss_fct, compute_random_on, random_layer, local_loss_fct, compute_local_on, local_layer, 
                     z1=None, z2=None, z1_ori=None, z2_ori=None, logits_ori=None, 
                     sdf1_all_pair_without_df_mask=None, sdf2_all_pair_without_df_mask=None):
        
        # Randomness
        loss_r = 0
        if random_layer == '1':
            all_z = [z1]
        elif random_layer == '2':
            all_z = [z2]
        elif random_layer == 'both':
            all_z = [z1, z2]
        else:
            raise NotImplementedError
        
        neg_size = data.df_mask.sum()
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=neg_size)

        if compute_random_on == 'edgeprob':       # Compute Randomness on edge probability
            
            for z in all_z:
                df_logits = model.decode(z, data.train_pos_edge_index[:, data.df_mask], neg_edge_index)
                loss_r += random_loss_fct(df_logits[:neg_size], df_logits[neg_size:])

        elif compute_random_on == 'nodeemb':
            for z in all_z:
                z_random_source, z_random_target = z[neg_edge_index[0]], z[neg_edge_index[1]]
                z_source, z_target = z[data.train_pos_edge_index[:, data.df_mask][0]], z[data.train_pos_edge_index[:, data.df_mask][1]]
                loss_r += (random_loss_fct(z_source, z_random_source) + random_loss_fct(z_target, z_random_target))

        elif compute_random_on == 'none':
            loss_r = None

        else:
            raise NotImplementedError


        # Local causality
        loss_l = 0
        if local_layer == '1':
            all_z = [z1]
            all_z_ori = [z1_ori]
            all_sdf_lower_triangular_mask = [sdf1_all_pair_without_df_mask]
            all_sdf_node_mask = [data.sdf_node_1hop_mask]
        elif local_layer == '2':
            all_z = [z2]
            all_z_ori = [z2_ori]
            all_sdf_lower_triangular_mask = [sdf2_all_pair_without_df_mask]
            all_sdf_node_mask = [data.sdf_node_2hop_mask]
        elif local_layer == 'both':
            all_z = [z1, z2]
            all_z_ori = [z1_ori, z2_ori]
            all_sdf_lower_triangular_mask = [sdf1_all_pair_without_df_mask, sdf2_all_pair_without_df_mask]
            all_sdf_node_mask = [data.sdf_node_1hop_mask, data.sdf_node_2hop_mask]
        else:
            raise NotImplementedError


        if compute_local_on == 'edgeprob':

            for z_ori, z, sdf_lower_triangular_mask in zip(all_z_ori, all_z, all_sdf_lower_triangular_mask):
                logits = (z @ z.t())[sdf_lower_triangular_mask].sigmoid()
                logits_ori = (z_ori @ z_ori.t())[sdf_lower_triangular_mask].sigmoid()

                loss_l += local_loss_fct(logits, logits_ori)
        
        elif compute_local_on == 'nodeemb':

            for z_ori, z, sdf_node_mask in zip(all_z_ori, all_z, all_sdf_node_mask):
                print(z_ori.shape, z.shape, sdf_node_mask.shape, sdf_node_mask.sum())
                loss_l += local_loss_fct(z_ori[sdf_node_mask], z[sdf_node_mask])

        elif compute_local_on == 'none':
            loss_l = None

        else:
            raise NotImplementedError


        if compute_random_on == 'none':
            loss = loss_l
        elif compute_local_on == 'none':
            loss = loss_r
        else:
            alpha = 0.5
            loss = alpha * loss_r + (1 - alpha) * loss_l

        return loss, loss_r, loss_l

    def train_fullbatch(self, model, data, retrain_data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to(self.device)
        # data = data.to(self.device)

        best_metric = 0
        patience = 0
        best_state_dict = model.state_dict()

        # '''Model naming convention: "gnndelete_random_mse_edgeprob_1_local_mse_edgeprob_1" '''
        # _, _, random_loss_fct, compute_random_on, random_layer, _, local_loss_fct, compute_local_on, local_layer = self.args.unlearning_model.split('_')
        # random_loss_fct = get_loss_fct(random_loss_fct)
        # local_loss_fct = get_loss_fct(local_loss_fct)

        # neg_size = 10

        # MI Attack before unlearning
        # if attack_model_all is not None:
        #     mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
        #     self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
        #     self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        # if attack_model_sub is not None:
        #     mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
        #     self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
        #     self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before

        # All node paris in S_Df without Df
        ## S_Df 1 hop all pair mask
        sdf1_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_1hop_mask], with_replacement=True).t()
        sdf1_all_pair_mask[idx[0], idx[1]] = True
        sdf1_all_pair_mask[idx[1], idx[0]] = True

        assert sdf1_all_pair_mask.sum().cpu() == data.sdf_node_1hop_mask.sum().cpu() * data.sdf_node_1hop_mask.sum().cpu()

        ## Remove Df itself
        sdf1_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1]] = False
        sdf1_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][1], data.train_pos_edge_index[:, data.df_mask][0]] = False

        ## S_Df 2 hop all pair mask
        sdf2_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        # idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_2hop_mask], with_replacement=True).t()
        # sdf2_all_pair_mask[idx[0], idx[1]] = True
        # sdf2_all_pair_mask[idx[1], idx[0]] = True
        sdf2_all_pair_mask[data.two_hop_edge_wo_df[0], data.two_hop_edge_wo_df[1]] = True
        sdf2_all_pair_mask[data.two_hop_edge_wo_df[1], data.two_hop_edge_wo_df[0]] = True

        # assert sdf2_all_pair_mask.sum().cpu() == data.sdf_node_2hop_mask.sum().cpu() * data.sdf_node_2hop_mask.sum().cpu()

        ## Remove Df itself
        # sdf2_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1]] = False
        # sdf2_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][1], data.train_pos_edge_index[:, data.df_mask][0]] = False

        ## Lower triangular mask
        idx = torch.tril_indices(data.num_nodes, data.num_nodes, -1)
        lower_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        lower_mask[idx[0], idx[1]] = True

        ## The final mask is the intersection
        sdf1_all_pair_without_df_mask = sdf1_all_pair_mask & lower_mask
        sdf2_all_pair_without_df_mask = sdf2_all_pair_mask & lower_mask

        # print(data.sdf_node_2hop_mask.sum())
        # print(sdf_all_pair_mask.nonzero())
        # print(data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1])
        # print('aaaaaaaaaaaa', data.sdf_node_2hop_mask.sum(), a, sdf_all_pair_mask.sum())
        # print('aaaaaaaaaaaa', lower_mask.sum())
        # print('aaaaaaaaaaaa', sdf_all_pair_without_df_mask.sum())
        # print('aaaaaaaaaaaa', data.sdf_node_2hop_mask.sum())
        # assert sdf_all_pair_without_df_mask.sum() == \
        #         data.sdf_node_2hop_mask.sum().cpu() * (data.sdf_node_2hop_mask.sum().cpu() - 1) // 2 - data.df_mask.sum().cpu()

        # Original node embeddings
        # with torch.no_grad():
        #     z1_ori, z2_ori = model.get_original_embeddings(data.x, data.train_pos_edge_index[:, data.dtrain_mask], return_all_emb=True)

        loss_fct = nn.MSELoss()

        for epoch in range(args.epochs):
            model.train()

            start_time = time.time()
            # _adj = copy.deepcopy(adj)
            _adj = torch.zeros(data.num_nodes, data.num_nodes)
            _adj[data.edge_index[0, data.sdf_mask.cpu()], data.edge_index[1, data.sdf_mask.cpu()]] = 1
            _adj[data.edge_index[1, data.sdf_mask.cpu()], data.edge_index[0, data.sdf_mask.cpu()]] = 1
            _adj_norm = utils.normalize(_adj.to(self.device) + torch.eye(data.num_nodes, device=self.device))

            _adj_prime = copy.deepcopy(_adj)
            _adj_prime[data.train_pos_edge_index[0, data.df_mask], data.train_pos_edge_index[1, data.df_mask]] = 0
            _adj_prime[data.train_pos_edge_index[1, data.df_mask], data.train_pos_edge_index[0, data.df_mask]] = 0
            _adj_prime_norm = utils.normalize(_adj_prime.to(self.device) + torch.eye(data.num_nodes, device=self.device))

            z = model(data.x.to(self.device), _adj_norm)
            z_wo_df = model(data.x.to(self.device), _adj_prime_norm)
            # z = model(data.x.to(self.device), data.train_pos_edge_index[:, data.sdf_mask])
            # z1, z2 = model(data.x, data.train_pos_edge_index[:, data.sdf_mask], return_all_emb=True)
            # print('current deletion weight', model.deletion1.deletion_weight.sum(), model.deletion2.deletion_weight.sum())
            # print('aaaaaa', z[data.sdf_node_2hop_mask].sum())

            # Effectiveness and Randomness
            # if data.df_mask.sum() == 8:
            #     neg_size = data.df_mask.sum() * 10
            # else:
            #     neg_size = data.df_mask.sum()

            neg_size = data.df_mask.sum() * 20
            # neg_size = data.df_mask.sum()
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=neg_size)

            df_logits = model.decode(z, data.train_pos_edge_index[:, data.df_mask], neg_edge_index)
            if neg_size != data.df_mask.sum():
                # print('11', df_logits[:data.df_mask.sum()].shape, df_logits.shape, data.df_mask.sum())
                loss_r = loss_fct(df_logits[:data.df_mask.sum()].repeat(20, 1), df_logits[data.df_mask.sum():])
            else:
                loss_r = loss_fct(df_logits[:neg_size], df_logits[neg_size:])
            # df_logits = model.decode(
            #     z, 
            #     data.train_pos_edge_index[:, data.df_mask].repeat(1, neg_size), 
            #     neg_edge_index).sigmoid()
            
            # loss_e = loss_fct(df_logits[:neg_size], df_logits[neg_size:])
            # print('df_logits', df_logits)
            # raise

            # Local causality
            if sdf2_all_pair_without_df_mask.sum() != 0:
                logits_sdf = (z_wo_df @ z_wo_df.t())[sdf2_all_pair_without_df_mask].sigmoid()
                loss_l = loss_fct(logits_sdf, logits_ori[sdf2_all_pair_without_df_mask].sigmoid())
                # print('local proba', logits_sdf.shape, logits_sdf, logits_ori[sdf2_all_pair_without_df_mask].sigmoid())
            
            else:
                loss_l = torch.tensor(0)

            alpha = 0.2
            loss = alpha * loss_r + (1 - alpha) * loss_l

            # loss, loss_r, loss_l = self.compute_loss(
            #     model, data, random_loss_fct, compute_random_on, random_layer, local_loss_fct, compute_local_on, local_layer,
            #     z1, z2, z1_ori, z2_ori, logits_ori, sdf1_all_pair_without_df_mask, sdf2_all_pair_without_df_mask)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            end_time = time.time()
            epoch_time = end_time - start_time

            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'loss_r': loss_r.item(),
                'loss_l': loss_l.item(),
                'train_time': epoch_time
            }

            # validation
            # valid_loss, val_auc = self.eval(model, retrain_data, 'val', device=self.device)
            # # print(f'Epoch: {epoch}, Train Loss: {loss.item():.4f}, Valid Loss: {valid_loss:.4f}, Valid AUC: {val_auc:.4f}')
            # if val_auc > best_metric:
            #     best_metric = val_auc
            #     best_epoch = epoch
            #     patience = 0

            #     best_state_dict = model.state_dict()
            # else:
            #     patience += 1
            #     if patience > 20:
            #         break

            # if (epoch + 1) % self.args.valid_freq == 0:
            #     valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')
            #     valid_log['epoch'] = epoch

            #     train_log = {
            #         'epoch': epoch,
            #         'train_loss': loss.item(),
            #         'train_loss_l': loss_l.item(),
            #         'train_loss_r': loss_r.item(),
            #         'train_time': epoch_time,
            #     }
                
            #     for log in [train_log, valid_log]:
            #         # wandb.log(log)
            #         msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
            #         tqdm.write(' | '.join(msg))
            #         self.trainer_log['log'].append(log)

            #     if dt_auc + df_auc > best_metric:
            #         best_metric = dt_auc + df_auc
            #         best_epoch = epoch

            #         print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
            #         ckpt = {
            #             'model_state': model.state_dict(),
            #             'optimizer_state': optimizer.state_dict(),
            #         }
            #         torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))

        # Save
        # ckpt = {
        #     'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
        #     'optimizer_state': optimizer.state_dict(),
        # }
        # torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))
        # model.load_state_dict(best_state_dict)


def parse_args():
    class Object(object):
        pass

    obj = Object()
    obj.unlearning_model = 'gnndelete'
    obj.gnn = 'gcn'
    obj.dataset = 'citeseer'
    obj.checkpoint_dir = 'checkpoint'
    obj.valid_freq = 10
    obj.epochs = 50


    return obj
    
class GNNDelete(Unlearn):

    def __init__(self, seed, data, features, adj, labels, config, device, model_type, epochs, verbose) -> None:
        super().__init__(seed, features, adj, labels, config, device, model_type, epochs, verbose)

        # self.num_epochs = epochs
        self.data = data
        self.edge_index = data.edge_index.to(self.device)

        self.model = GCNDelete(
            nfeat=self.features.shape[1],
            nhid=self.config['nhid'],
            nclass=int(self.labels.max().item() + 1),
            droput=self.config['dropout'],
        ).to(self.device)
        
        # self.model = GCN(nfeat=self.features.shape[1], 
        #             nhid=self.config['nhid'], 
        #             nclass=int(self.labels.max().item() + 1), 
        #             dropout=self.config['dropout']).to(self.device)

        # sdf_node_1hop = torch.zeros(self.num_nodes, dtype=torch.bool)
        # sdf_node_2hop = torch.zeros(self.num_nodes, dtype=torch.bool)
        # sdf_node_1hop[one_hop_edge.flatten().unique()] = True
        # sdf_node_2hop[two_hop_edge.flatten().unique()] = True
        # self.model = GCNDelete(in_dim=self.features.shape[1],
        #                        hidden_dim=self.config['nhid'],
        #                        out_dim=int(self.labels.max().item() + 1),
        #                        mask_1hop=None, mask_2hop=None).to(self.device)
        
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
            # self.model.train()
            # output = F.log_softmax(self.model(self.features, self.edge_index), dim=1)
            # loss_train = F.nll_loss(output[self.idx_train], _labels_train)

            # loss_train.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            self.model.eval()
            with torch.no_grad():
                output = self.model(self.features, self.adj_norm)[self.idx_val]
                loss_val = criterion(output, _labels_val)
                # output = F.log_softmax(self.model(self.features, self.edge_index), dim=1)[self.idx_val]
                # loss_val = F.nll_loss(output, _labels_val)

            if loss_val < best_valid_loss:
                best_valid_loss = loss_val
                trial_count = 0 
                best_model_state = self.model.state_dict()
            else:
                trial_count += 1
                if trial_count > 10:
                    break

            # print(f'Epoch {e+1:03d}: train loss {loss_train.item():.4f}, val loss {loss_val.item():.4f}')
        
        self.model.load_state_dict(best_model_state)

        # evaluate the learned model
        return self._evaluate(self.model, self.adj_norm)
    
    def posterior(self, indices=None, use_retrained=False):
        model = self.retrain_model if use_retrained else self.model
        edge_index = self.retrain_edge_index if use_retrained else self.edge_index

        model.eval()
        with torch.no_grad():
            outputs = model(self.features, edge_index)

        if indices is not None:
            return outputs[indices]
        else:
            return outputs
    
    def unlearn(self, edges_to_forget):
        self.retrain_data = copy.deepcopy(self.data)
        self.retrain_data.remove_edges(edges_to_forget)
        self.retrain_edge_index = self.retrain_data.edge_index.to(self.device)

        self.adj_prime = copy.deepcopy(self.adj)
        delete_edge_index = to_undirected(torch.tensor(edges_to_forget).t())
        self.adj_prime[delete_edge_index[0], delete_edge_index[1]] = 0
        self.adj_prime_norm = utils.normalize(self.adj_prime + torch.eye(self.adj_prime.shape[0]).to(self.device))

        # print('edges:', edges_to_forget)
        # construct the mask of edge index
        # _edge_index = utils.to_directed(self.edge_index)
        deletion_mask = torch.zeros(self.edge_index.shape[1], dtype=torch.bool)
        for u, v in edges_to_forget:
            _match = torch.eq(self.edge_index, torch.tensor([u, v], device=self.device).view(2, -1))
            uv_idx = _match[0] & _match[1]
            _match = torch.eq(self.edge_index, torch.tensor([v, u], device=self.device).view(2, -1))
            vu_idx = _match[0] & _match[1]
            if uv_idx.sum() == 1 and vu_idx.sum() == 1:
                deletion_mask[torch.nonzero(uv_idx).squeeze()] = True
                deletion_mask[torch.nonzero(vu_idx).squeeze()] = True
            else:
                raise ValueError(f'Invalid edge index, multiple e: ({u},{v}), {uv_idx.sum()}, {vu_idx.sum()}') 
        # print('edges 2:', _edge_index[:, deletion_mask])
        _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
            self.edge_index[:, deletion_mask].flatten().unique(),
            2, self.edge_index, num_nodes=self.num_nodes
        )

        _, two_hop_edge_wo_df, _, two_hop_mask_wo_df = k_hop_subgraph(
            self.edge_index[:, deletion_mask].flatten().unique(),
            2, self.retrain_edge_index, num_nodes=self.num_nodes
        )
        two_hop_mask = two_hop_mask.to('cpu')
        self.data.sdf_mask = two_hop_mask

        _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
            self.edge_index[:, deletion_mask].flatten().unique(),
            1, self.edge_index, num_nodes=self.num_nodes
        )
        sdf_node_1hop = torch.zeros(self.num_nodes, dtype=torch.bool)
        sdf_node_2hop = torch.zeros(self.num_nodes, dtype=torch.bool)
        sdf_node_1hop[one_hop_edge.flatten().unique()] = True
        sdf_node_2hop[two_hop_edge.flatten().unique()] = True

        self.data.sdf_node_1hop_mask = sdf_node_1hop
        self.data.sdf_node_2hop_mask = sdf_node_2hop

        # train_pos_edge_index, [deletion_mask, two_hop_mask] = to_undirected(_edge_index.cpu(), [deletion_mask, two_hop_mask])
        train_pos_edge_index = self.edge_index
        two_hop_mask = two_hop_mask.bool().to(self.device)
        # deletion_mask = deletion_mask.bool()
        dr_mask = ~deletion_mask
        # print('edges 3:', train_pos_edge_index[:, deletion_mask])

        self.data.train_pos_edge_index = train_pos_edge_index
        self.data.retrain_pos_edge_index = self.retrain_edge_index
        self.retrain_data.train_pos_edge_index = train_pos_edge_index
        # self.edge_index = train_pos_edge_index
        self.data.sdf_mask = two_hop_mask
        self.data.dr_mask = dr_mask
        self.retrain_data.dr_mask = dr_mask
        self.data.df_mask = deletion_mask
        self.data.two_hop_edge_wo_df = two_hop_edge_wo_df
        # print('1111', self.data.df_mask.sum())
        
        # self.data.train_pos_edge_index = self.edge_index
        # self.data.sdf_mask = _two_hop_mask

        # self.retrain = GCNDelete(in_dim=self.features.shape[1],
        #                          hidden_dim=self.config['nhid'],
        #                          out_dim=int(self.labels.max().item() + 1),
        #                          mask_1hop=sdf_node_1hop,
        #                          mask_2hop=sdf_node_2hop) 
        self.retrain = GCNDeletion(
            nfeat=self.features.shape[1],
            nhid=self.config['nhid'],
            nclass=int(self.labels.max().item() + 1),
            dropout=self.config['dropout'],
            mask_1hop=sdf_node_1hop,
            mask_2hop=sdf_node_2hop
        )
        self.retrain.load_state_dict(self.model.state_dict(), strict=False)
        self.retrain = self.retrain.to(self.device)
        self.retrain.eval()
        with torch.no_grad():
            z = self.model(self.features, self.adj_norm)
            logits_ori = z @ z.t()

        self.model.train()
        _adj = torch.zeros_like(self.adj_prime)
        _adj[self.edge_index[0, self.data.sdf_mask.cpu()], self.edge_index[1, self.data.sdf_mask.cpu()]] = 1
        _adj_norm = utils.normalize(_adj.to(self.device) + torch.eye(self.num_nodes, device=self.device))
        z = self.model(self.features, _adj_norm)

        parameters_to_optimize = [
            {'params': [p for n, p in self.retrain.named_parameters() if 'del' in n], 'weight_decay': 1E-5},
        ]
        optimizer = torch.optim.Adam(parameters_to_optimize, lr=0.001, weight_decay=1E-5)

        _args = parse_args()
        trainer = GNNDeleteTrainer(_args, device=self.device)
        trainer.train(self.retrain, self.data, self.retrain_data, optimizer, _args, logits_ori=logits_ori, attack_model_all=None, attack_model_sub=None)

        # evaluate the learned model
        return self._evaluate(self.retrain, self.adj_prime_norm)


    def predict(self, target_nodes, use_retrained=False, return_posterior=False):
        model = self.retrain if use_retrained else self.model
        edge_index = self.retrain_edge_index if use_retrained else self.edge_index
        adj_norm = self.adj_prime_norm if use_retrained else self.adj_norm

        model.eval()
        with torch.no_grad():
            outputs = model(self.features, adj_norm)[target_nodes]
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        if return_posterior:
            return y_pred, outputs.cpu().detach()
        else:
            return y_pred
        
    def parameters(self, use_retrained=False):
        model = self.retrain if use_retrained else self.model
        return [p for n, p in model.named_parameters() if 'gc' in n], [p for n, p in model.named_parameters() if 'del' in n]

    def _evaluate(self, model, adj):
        model.eval()
        with torch.no_grad():
            outputs = model(self.features, adj)[self.idx_test]
        loss = F.cross_entropy(outputs, self.labels[self.idx_test]) 
        y_preds = outputs.argmax(dim=1)
        y_true = self.labels[self.idx_test].cpu().numpy()
        result = classification_report(y_true, y_preds.cpu().numpy(), output_dict=True, zero_division=0)
        result['loss'] = loss.cpu().item()
        return result