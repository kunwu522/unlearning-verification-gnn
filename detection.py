""" 
    Three detection methods
    1. LinkPred
    2. OutlierD
    Characterizing Malicious Edges targeting on Graph Neural Networks, Xu et al., 2019

    3. ProximityDetect
    COMPARING AND DETECTING ADVERSARIAL ATTACKS FOR GRAPH DEEP LEARNING, Zhang et al., 2019

    Author: hiding for anonymity
    
"""
import os
import time
import math
import copy
import random
import pickle
import argparse
import statistics
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from torch_geometric.utils import to_undirected, k_hop_subgraph

from model.layers import GraphConvolution
from model.gcn import GCN
import utils

class LinkPred(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, activation=True, bias=True) -> None:
        super(LinkPred, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, bias=bias)
        self.gc2 = GraphConvolution(nhid, nclass, bias=bias)
        self.dropout = dropout
        self.activation = activation

        self.w_edge = nn.Linear(nclass, nclass, bias=False)
    
    def forward(self, u, v, x, adj):
        if self.activation:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        p_uv = torch.sigmoid(self.w_edge(x[u]).mm(x[v].t()))
        return p_uv


class LinkPredDetector(object):
    def __init__(self, data, v_star, e_star, device):
        # print('nettack:', e_star)
        self.data = data
        self.v_star = v_star
        self.e_star = e_star
        self.device = device

    def _train_gcn(self, adv_data):
        train_links = random.sample(self.data.edges, int(len(self.data.edges) * 0.5))

        adj = adv_data.adjacency_matrix().to(self.device)
        adj_tilde = torch.eye(adj.shape[0]).to(self.device) + adj
        adj_norm = utils.normalize(adj_tilde)

        # randomly sample edges 
        non_links = torch.where(adj_tilde.to_dense() == 0)
        non_links = torch.stack([non_links[0], non_links[1]], dim=0).t().tolist()
        non_links = random.sample(non_links, len(train_links))

        train_links = np.array(train_links + non_links)
        train_labels = np.array([1] * len(non_links) + [0] * len(non_links))
        rand_indices = np.random.permutation(len(train_links))
        train_links = train_links[rand_indices]
        train_labels = train_labels[rand_indices]

        # train_links = torch.tensor(train_links).to(self.device)
        
        # Train GCN to obtain embeddings
        model = LinkPred(adv_data.num_features, 32, 16, dropout=0.5).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        # for e in tqdm(range(20), desc='Training LinkPred'):
        for e in range(20):
            model.train()
            for batch in range(0, len(train_links), 1024):
                batch_links = train_links[batch: batch+1024]
                batch_labels = train_labels[batch: batch+1024]
                batch_links = torch.tensor(batch_links).t().to(self.device)
                batch_labels = torch.tensor(batch_labels).float().to(self.device)
                output = model(batch_links[0], batch_links[1], adv_data.x.to(self.device), adj_norm)
                loss = criterion(output.diag(), batch_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # print(f'Epoch {e}: Loss {loss.item():.4f}')
        model.eval()
        return model
    
    def detect_all(self):
        challenge_edges = [tuple(e) for ee in self.e_star for e in ee]
        challenge_edges = list(set(challenge_edges))

        adv_data = copy.deepcopy(self.data)
        adv_data.add_edges(to_undirected(torch.tensor(challenge_edges).t()))
        model = self._train_gcn(adv_data)

        adj = adv_data.adjacency_matrix().to(self.device)
        adj_tilde = torch.eye(adj.shape[0]).to(self.device) + adj
        adj_norm = utils.normalize(adj_tilde)
        directed_edge_index = utils.to_directed(adv_data.edge_index)

        # Compute the score for each edge
        # with torch.no_grad():
        #     test_links = torch.tensor(challenge_edges).t().to(self.device).long()
        #     output = model(test_links[0], test_links[1], adv_data.x.to(self.device), adj_norm).diag()
        #     scores = output.cpu().detach().numpy()
        # print('Method: Link Prediction')
        # print('number of edges:', len(challenge_edges))
        # print('scores:', scores)
        # return np.sum(scores < 0.5) / len(scores)

        with torch.no_grad():
            output = model(directed_edge_index[0], directed_edge_index[1], adv_data.x.to(self.device), adj_norm).diag()
            scores = output.cpu().detach().numpy()
            sorted_indices = np.argsort(scores)
            top_k_edges = directed_edge_index.t()[sorted_indices[:len(challenge_edges)]].numpy()

        top_k_edges_list = list(map(tuple, top_k_edges.tolist()))
        detection = []
        for e in challenge_edges:
            if e in top_k_edges_list or (e[1], e[0]) in top_k_edges_list:
                detection.append(1)
            else:
                detection.append(0)
        ratio_edge = np.sum(detection) / len(detection)
        return ratio_edge >= 0.2
        # return np.sum(detection) / len(challenge_edges)

        # with torch.no_grad():
        #     output = model(directed_edge_index[0], directed_edge_index[1], adv_data.x.to(self.device), adj_norm).diag()
        #     scores = output.cpu().detach().numpy()
        #     sorted_indices = np.argsort(scores)
        #     top_k_edges = directed_edge_index.t()[sorted_indices[:len(challenge_edges)]].numpy()
        nodes, counts = np.unique(top_k_edges.flatten(), return_counts=True)
        malicious_nodes = nodes[counts >= 0.5 * len(challenge_edges)]

        detection = []
        escaped_nodes = list(set(self.v_star) - set(malicious_nodes.tolist()))
        for n in malicious_nodes:
            if n in self.v_star:
                detection.append(1)
            else:
                detection.append(0)
        ratio_node = np.sum(detection) / len(self.v_star)

        return ratio_edge, ratio_node, escaped_nodes

    def detect_one(self, es):
        adv_data = copy.deepcopy(self.data)
        adv_data.add_edges(to_undirected(torch.tensor(es).t()))
        model = self._train_gcn(adv_data)

        adj = adv_data.adjacency_matrix().to(self.device)
        adj_tilde = torch.eye(adj.shape[0]).to(self.device) + adj
        adj_norm = utils.normalize(adj_tilde)

        with torch.no_grad():
            test_links = torch.tensor(es).t().to(self.device).long()
            output = model(test_links[0], test_links[1], adv_data.x.to(self.device), adj_norm).diag()
            scores = output.cpu().detach().numpy()
        
        return np.sum(scores < 0.5) > len(es) * 0.5
    
    def detect(self, v, es):
        challenge_edges = [tuple(e) for e in es]
        challenge_edges = list(set(challenge_edges))

        adv_data = copy.deepcopy(self.data)
        adv_data.add_edges(to_undirected(torch.tensor(challenge_edges).t()))
        model = self._train_gcn(adv_data)

        adj = adv_data.adjacency_matrix().to(self.device)
        adj_tilde = torch.eye(adj.shape[0]).to(self.device) + adj
        adj_norm = utils.normalize(adj_tilde)
        directed_edge_index = utils.to_directed(adv_data.edge_index)

        # Compute the score for each edge
        # with torch.no_grad():
        #     test_links = torch.tensor(challenge_edges).t().to(self.device).long()
        #     output = model(test_links[0], test_links[1], adv_data.x.to(self.device), adj_norm).diag()
        #     scores = output.cpu().detach().numpy()
        # print('Method: Link Prediction')
        # print('number of edges:', len(challenge_edges))
        # print('scores:', scores)
        # return np.sum(scores < 0.5) / len(scores)

        with torch.no_grad():
            output = model(directed_edge_index[0], directed_edge_index[1], adv_data.x.to(self.device), adj_norm).diag()
            scores = output.cpu().detach().numpy()
            sorted_indices = np.argsort(scores)
            top_k_edges = directed_edge_index.t()[sorted_indices[:len(challenge_edges)]].numpy()

        top_k_edges_list = list(map(tuple, top_k_edges.tolist()))
        detection = []
        for e in challenge_edges:
            if e in top_k_edges_list or (e[1], e[0]) in top_k_edges_list:
                detection.append(1)
            else:
                detection.append(0)
        ratio_edge = np.sum(detection) / len(detection)
        # return np.sum(detection) / len(challenge_edges)

        # with torch.no_grad():
        #     output = model(directed_edge_index[0], directed_edge_index[1], adv_data.x.to(self.device), adj_norm).diag()
        #     scores = output.cpu().detach().numpy()
        #     sorted_indices = np.argsort(scores)
        #     top_k_edges = directed_edge_index.t()[sorted_indices[:len(challenge_edges)]].numpy()
        nodes, counts = np.unique(top_k_edges.flatten(), return_counts=True)
        malicious_nodes = nodes[counts >= 5]
        if v in malicious_nodes:
            return True
        else:
            return False

        # detection = []
        # escaped_nodes = list(set(self.v_star) - set(malicious_nodes.tolist()))
        # for n in malicious_nodes:
        #     if n in self.v_star:
        #         detection.append(1)
        #     else:
        #         detection.append(0)
        # ratio_node = np.sum(detection) / len(self.v_star)

        # return ratio_edge, ratio_node, escaped_nodes
        # detection_ratio_list = []
        # for challenge_edges in tqdm(self.e_star, desc='LinkPred detecting'):
        #     adv_data = copy.deepcopy(self.data)
        #     adv_data.add_edges(to_undirected(torch.tensor(challenge_edges).t()))
        #     model = self._train_gcn(adv_data)

        #     adj = adv_data.adjacency_matrix().to(self.device)
        #     adj_tilde = torch.eye(adj.shape[0]).to(self.device) + adj
        #     adj_norm = utils.normalize(adj_tilde)

        #     # Compute the score for each edge
        #     # with torch.no_grad():
        #     #     test_links = torch.tensor(challenge_edges).t().to(self.device).long()
        #     #     output = model(test_links[0], test_links[1], adv_data.x.to(self.device), adj_norm).diag()
        #     #     scores = output.cpu().detach().numpy()
        #     # detection_ratio_list.append(np.sum(scores < 0.5) / len(scores))

        #     with torch.no_grad():
        #         output = model(adv_data.edge_index[0], adv_data.edge_index[1], adv_data.x.to(self.device), adj).diag()
        #         scores = output.cpu().detach().numpy()
        #         sorted_indices = np.argsort(scores)
        #         top_k_edges = adv_data.edge_index.t()[sorted_indices[:len(challenge_edges)]]
        #         top_k_edges = list(map(tuple, top_k_edges.tolist()))
        #         detection = []
        #         for e in challenge_edges:
        #             if e in top_k_edges or (e[1], e[0]) in top_k_edges:
        #                 detection.append(1)
        #             else:
        #                 detection.append(0)
        #         detection_ratio_list.append(np.sum(detection) / len(detection))
        # return np.mean(detection_ratio_list)
    

class OutlierDetector(object):
    def __init__(self, data, v_star, e_star) -> None:
        self.data = data
        self.v_star = v_star
        self.e_star = e_star

    def _compute_feautres(self, adv_data, v):
        neighbors = list(adv_data.adj_list[v])
        _x = np.zeros(5)
        uni_classes, counts = np.unique(adv_data.y[neighbors].cpu().numpy(), return_counts=True)
        sorted_counts = np.sort(counts)[::-1]
        _x[0] = len(uni_classes)
        _x[1] = np.mean(counts) if len(counts) > 0 else 0
        _x[2] = sorted_counts[0] if len(sorted_counts) > 0 else 0
        _x[3] = sorted_counts[1] if len(sorted_counts) > 1 else 0
        _x[4] = np.std(counts) if len(counts) > 0 else 0
        return _x

    def detect(self, v, es):
        """ 1. number of different clases in the neighbours
            2. average appearance time of each class in the neighbours
            3. appeearance time of the most frequently appeared class in the neighbours
            4. appearance time of the second most frequently appeared class in the neighbour
            5. standard deviation of the appearance time of each class in the neighbours
        """
        challenge_edges = [tuple(e) for e in es]
        challenge_edges = list(set(challenge_edges))

        adv_data = copy.deepcopy(self.data)
        adv_data.add_edges(to_undirected(torch.tensor(challenge_edges).t()))

        # all edges
        directed_edge_index = utils.to_directed(adv_data.edge_index)
        num_edges = directed_edge_index.shape[1]
        train_features = np.zeros((num_edges, 10))
        for i, (a, b) in enumerate(directed_edge_index.t().cpu().tolist()):
            train_features[i, :5] = self._compute_feautres(adv_data, a)
            train_features[i, 5:] = self._compute_feautres(adv_data, b)

        # Train a one-class SVM
        clf = svm.OneClassSVM(gamma='scale').fit(train_features)

        # all edges with scores
        # Compute the score for each edge
        edges = [tuple(e) for e in directed_edge_index.t().cpu().tolist()]
        test_features = np.zeros((len(edges), 10)) 
        challenge_edges_indices = []
        checked_edges = set()
        for i, (a, b) in enumerate(edges):
            if (a, b) in checked_edges or (b, a) in checked_edges:
                continue
            test_features[i, :5] = self._compute_feautres(adv_data, a)
            test_features[i, 5:] = self._compute_feautres(adv_data, b)
            if (a, b) in challenge_edges or (b, a) in challenge_edges:
                challenge_edges_indices.append(i)
            checked_edges.add((a, b))
        scores = clf.score_samples(test_features)
        # print('challenge edges scores:', scores[challenge_edges_indices])
        # print('the number of scores that are 1:', np.sum(scores == 1))
        # return np.sum(scores == 1) / len(scores)
        sorted_indices = np.argsort(scores)[::-1]
        top_k_edges = directed_edge_index.t().numpy()[sorted_indices[:len(challenge_edges)]]

        # only challenge edges
        # test_features = np.zeros((len(challenge_edges), 10))
        # for i, (a, b) in enumerate(challenge_edges):
        #     test_features[i, :5] = self._compute_feautres(adv_data, a)
        #     test_features[i, 5:] = self._compute_feautres(adv_data, b)
        # preds = clf.predict(test_features)
        # top_k_edges = np.array(challenge_edges)[preds == 1]

        top_k_edges_list = list(map(tuple, top_k_edges.tolist()))
        detection = []
        for e in challenge_edges:
            if e in top_k_edges_list or (e[1], e[0]) in top_k_edges_list:
                detection.append(1)
            else:
                detection.append(0)
        ratio_edge = np.sum(detection) / len(detection)
        return ratio_edge >= 0.5
        # return np.sum(detection) / len(challenge_edges)

        # with torch.no_grad():
        #     output = model(directed_edge_index[0], directed_edge_index[1], adv_data.x.to(self.device), adj_norm).diag()
        #     scores = output.cpu().detach().numpy()
        #     sorted_indices = np.argsort(scores)
        #     top_k_edges = directed_edge_index.t()[sorted_indices[:len(challenge_edges)]].numpy()
        nodes, counts = np.unique(top_k_edges.flatten(), return_counts=True)
        malicious_nodes = nodes[counts >= 5]
        if v in malicious_nodes:
            return True
        else:
            return False


        # detection = []
        # escaped_nodes = list(set(self.v_star) - set(malicious_nodes.tolist()))
        # for n in malicious_nodes:
        #     if n in self.v_star:
        #         detection.append(1)
        #     else:
        #         detection.append(0)
        # ratio_node = np.sum(detection) / len(self.v_star)
        return ratio_edge, ratio_node, escaped_nodes
        # detection_ratio_list = []
        # for challenge_edges in tqdm(self.e_star, desc='Outlier detecting'):
        #     adv_data = copy.deepcopy(self.data)
        #     adv_data.add_edges(to_undirected(torch.tensor(challenge_edges).t()))

        #     directed_edge_index = utils.to_directed(adv_data.edge_index)
        #     num_edges = directed_edge_index.shape[1]
        #     train_features = np.zeros((num_edges, 10))
        #     for i, (a, b) in enumerate(directed_edge_index.t().cpu().tolist()):
        #         train_features[i, :5] = self._compute_feautres(adv_data, a)
        #         train_features[i, 5:] = self._compute_feautres(adv_data, b)

        #     # Train a one-class SVM
        #     clf = svm.OneClassSVM(gamma='scale').fit(train_features)

        #     # Compute the score for each edge
        #     test_features = np.zeros((len(challenge_edges), 10)) 
        #     for i, (a, b) in enumerate(challenge_edges):
        #         test_features[i, :5] = self._compute_feautres(adv_data, a)
        #         test_features[i, 5:] = self._compute_feautres(adv_data, b) 
        #     scores = clf.predict(test_features)
        #     detection_ratio_list.append(np.sum(scores == 1) / len(scores))
        # return np.mean(detection_ratio_list)
    

    def detect_one(self, es):
        adv_data = copy.deepcopy(self.data)
        adv_data.add_edges(to_undirected(torch.tensor(es).t()))

        directed_edge_index = utils.to_directed(adv_data.edge_index)
        num_edges = directed_edge_index.shape[1]
        train_features = np.zeros((num_edges, 10))
        for i, (a, b) in enumerate(directed_edge_index.t().cpu().tolist()):
            train_features[i, :5] = self._compute_feautres(adv_data, a)
            train_features[i, 5:] = self._compute_feautres(adv_data, b)
        for i, (a, b) in enumerate(es):
            train_features[i, :5] = self._compute_feautres(adv_data, a)
            train_features[i, 5:] = self._compute_feautres(adv_data, b)
        
        clf = svm.OneClassSVM(gamma='scale').fit(train_features) 
        # Compute the score for each edge
        test_features = np.zeros((len(es), 10)) 
        for i, (a, b) in enumerate(es):
            test_features[i, :5] = self._compute_feautres(adv_data, a)
            test_features[i, 5:] = self._compute_feautres(adv_data, b) 
        scores = clf.predict(test_features)
        
        return np.sum(scores == 1) > len(es) * 0.5

    def detect_all(self):
        """ 1. number of different clases in the neighbours
            2. average appearance time of each class in the neighbours
            3. appeearance time of the most frequently appeared class in the neighbours
            4. appearance time of the second most frequently appeared class in the neighbour
            5. standard deviation of the appearance time of each class in the neighbours
        """
        challenge_edges = [tuple(e) for ee in self.e_star for e in ee]
        challenge_edges = list(set(challenge_edges))

        adv_data = copy.deepcopy(self.data)
        adv_data.add_edges(to_undirected(torch.tensor(challenge_edges).t()))

        # all edges
        directed_edge_index = utils.to_directed(adv_data.edge_index)
        num_edges = directed_edge_index.shape[1]
        train_features = np.zeros((num_edges, 10))
        for i, (a, b) in enumerate(directed_edge_index.t().cpu().tolist()):
            train_features[i, :5] = self._compute_feautres(adv_data, a)
            train_features[i, 5:] = self._compute_feautres(adv_data, b)

        # Train a one-class SVM
        clf = svm.OneClassSVM(gamma='scale').fit(train_features)

        # all edges with scores
        # Compute the score for each edge
        edges = [tuple(e) for e in directed_edge_index.t().cpu().tolist()]
        test_features = np.zeros((len(edges), 10)) 
        challenge_edges_indices = []
        checked_edges = set()
        for i, (a, b) in enumerate(edges):
            if (a, b) in checked_edges or (b, a) in checked_edges:
                continue
            test_features[i, :5] = self._compute_feautres(adv_data, a)
            test_features[i, 5:] = self._compute_feautres(adv_data, b)
            if (a, b) in challenge_edges or (b, a) in challenge_edges:
                challenge_edges_indices.append(i)
            checked_edges.add((a, b))
        scores = clf.score_samples(test_features)
        # print('challenge edges scores:', scores[challenge_edges_indices])
        # print('the number of scores that are 1:', np.sum(scores == 1))
        # return np.sum(scores == 1) / len(scores)
        sorted_indices = np.argsort(scores)[::-1]
        top_k_edges = directed_edge_index.t().numpy()[sorted_indices[:len(challenge_edges)]]

        # only challenge edges
        # test_features = np.zeros((len(challenge_edges), 10))
        # for i, (a, b) in enumerate(challenge_edges):
        #     test_features[i, :5] = self._compute_feautres(adv_data, a)
        #     test_features[i, 5:] = self._compute_feautres(adv_data, b)
        # preds = clf.predict(test_features)
        # top_k_edges = np.array(challenge_edges)[preds == 1]

        top_k_edges_list = list(map(tuple, top_k_edges.tolist()))
        detection = []
        for e in challenge_edges:
            if e in top_k_edges_list or (e[1], e[0]) in top_k_edges_list:
                detection.append(1)
            else:
                detection.append(0)
        ratio_edge = np.sum(detection) / len(detection)
        return ratio_edge >= 0.2
        # return np.sum(detection) / len(challenge_edges)

        # with torch.no_grad():
        #     output = model(directed_edge_index[0], directed_edge_index[1], adv_data.x.to(self.device), adj_norm).diag()
        #     scores = output.cpu().detach().numpy()
        #     sorted_indices = np.argsort(scores)
        #     top_k_edges = directed_edge_index.t()[sorted_indices[:len(challenge_edges)]].numpy()
        nodes, counts = np.unique(top_k_edges.flatten(), return_counts=True)
        malicious_nodes = nodes[counts > 0.5 * len(challenge_edges)]

        detection = []
        escaped_nodes = list(set(self.v_star) - set(malicious_nodes.tolist()))
        for n in malicious_nodes:
            if n in self.v_star:
                detection.append(1)
            else:
                detection.append(0)
        ratio_node = np.sum(detection) / len(self.v_star)
        return ratio_edge, ratio_node, escaped_nodes
        # detection = []
        # for e in challenge_edges:
        #     if e in top_k_edges or (e[1], e[0]) in top_k_edges:
        #         detection.append(1)
        #     else:
        #         detection.append(0) 
        # return np.sum(detection) / len(detection)
    
class ProximityDetector(object):
    def __init__(self, data, v_star, e_star, device) -> None:
        self.data = data
        self.v_star = v_star
        self.e_star = e_star
        self.device = device
        
        self.adj = self.data.adjacency_matrix().to_dense()
        self.adj_tilde = torch.eye(self.adj.shape[0]) + self.adj
        self.adj_norm = utils.normalize(self.adj_tilde)

        self.surrogate = GCN(self.data.num_features, 16, self.data.num_classes, dropout=0.5, bias=False, activation=False).to(self.device)
        self._train_gcn(self.surrogate)

        self.t1, self.t2 = self._compute_threshold(self.surrogate)

    def _kl_divergence(self, p, q):
        kl = np.where(p != 0, p * np.log(p / q), 0)
        kl = np.where(np.isnan(kl), 0, kl)
        kl = np.where(np.isinf(kl), 0, kl)
        return np.sum(kl)

    def _compute_threshold(self, model):
        x = self.data.x.to(self.device)
        posterior = F.softmax(model(x, self.adj_norm.to(self.device)), dim=1)

        train_nodes = self.data.train_set.nodes.tolist()
        prox1 = np.zeros(len(train_nodes))
        prox2 = np.zeros(len(train_nodes))
        for i, v in enumerate(train_nodes):
            neighbors = list(self.data.adj_list[v])
            if len(neighbors) <= 1:
                continue
            
            _prox1, _prox2 = 0, 0
            for u in neighbors:
                _prox1 += self._kl_divergence(posterior[v].detach().cpu().numpy(), posterior[u].detach().cpu().numpy())
                # n_u = list(self.data.adj_list[u])
                for k in neighbors:
                    _prox2 += self._kl_divergence(posterior[u].detach().cpu().numpy(), posterior[k].detach().cpu().numpy())
            prox1[i] = _prox1 / len(neighbors)
            prox2[i] = _prox2 / (len(neighbors) * (len(neighbors) - 1))

        print('statistics of prox1:', np.mean(prox1), np.max(prox1), np.min(prox1))
        print('statistics of prox2:', np.mean(prox2), np.max(prox2), np.min(prox2))
        mean1, std_dev1 = norm.fit(prox1)
        mean2, std_dev2 = norm.fit(prox2)
        target_false_positive_rate = 0.15
        # t1 = norm.isf(target_false_positive_rate, loc=mean1, scale=std_dev1)
        # t2 = norm.isf(target_false_positive_rate, loc=mean2, scale=std_dev2)
        t1 = norm.ppf(1 - target_false_positive_rate, loc=mean1, scale=std_dev1)
        t2 = norm.ppf(1 - target_false_positive_rate, loc=mean2, scale=std_dev2)
        return t1, t2

    def _train_gcn(self, model):
        train_loader = DataLoader(self.data.train_set, batch_size=512, shuffle=True)
        valid_loader = DataLoader(self.data.valid_set, batch_size=1024, shuffle=False)
        # edge_index = self.data.edge_index.to(device)
        adj = torch.sparse_coo_tensor(self.data.edge_index.cpu(), torch.ones(self.data.edge_index.size(1)), 
                                      size=(self.data.num_nodes, self.data.num_nodes))
        adj = utils.normalize(torch.eye(self.data.num_nodes) + adj).to_dense()

        adj = adj.to(self.device)
        x = self.data.x.to(self.device)
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()

        best_valid_loss = math.inf
        best_epoch = 0
        trial_count = 0
        best_model = None

        for e in range(1, 100 + 1):
            train_loss = 0.
            model.train()
            
            # iterator = tqdm(train_loader, f'  Epoch {e}') 
            for nodes, y in train_loader:
                nodes, y = nodes.to(self.device), y.to(self.device)

                model.zero_grad()
                output = model(x, adj)
                # output = self.model(x, edge_index)
                loss = criterion(output[nodes], y)
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().item()
            
            train_loss /= len(train_loader)
            
            valid_loss = 0.
            model.eval()
            with torch.no_grad():
                for nodes, y in valid_loader:
                    nodes = nodes.to(self.device)
                    y = y.to(self.device)
                    outputs = model(x, adj)
                    # outputs = self.model(x, edge_index)
                    loss = criterion(outputs[nodes], y)
                    valid_loss += loss.cpu().item()
            valid_loss /= len(valid_loader)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                trial_count = 0
                best_epoch = e
                best_model = copy.deepcopy(model)
            else:
                trial_count += 1
                if trial_count > 10:
                    print(f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                    break
        
        model = best_model
        model.eval()
        truth, preds = [], []
        with torch.no_grad():
            for nodes, y in valid_loader:
                nodes = nodes.to(self.device)
                # y = y.to(self.device)
                outputs = model(x, adj)
                preds.append(torch.argmax(outputs[nodes], dim=1).cpu().numpy())
                truth.append(y.numpy())
        preds = np.concatenate(preds)
        truth = np.concatenate(truth)
        acc = np.sum(preds == truth) / len(truth)
        print(f'  Validation accuracy: {acc:.4f}')
        model.cpu()

    def _compute_score(self, model, v, adv_data):
        # adv_data = copy.deepcopy(self.data)
        # adv_data.add_edges(to_undirected(torch.tensor(es).t()))
        adv_adj = adv_data.adjacency_matrix().to_dense()
        adv_adj_tilde = torch.eye(adv_adj.shape[0]) + adv_adj
        adv_adj_norm = utils.normalize(adv_adj_tilde).to(self.device)

        posterior = F.softmax(model(adv_data.x.to(self.device), adv_adj_norm), dim=1).detach().cpu().numpy()
        neighbors = list(adv_data.adj_list[v])
        num_neighbors = len(neighbors) if len(neighbors) > 0 else 0.00001
        _prox1, _prox2 = 0, 0
        for u in neighbors:
            _prox1 += self._kl_divergence(posterior[v], posterior[u])
            # n_u = list(adv_data.adj_list[u])
            for k in neighbors:
                _prox2 += self._kl_divergence(posterior[u], posterior[k])
        prox1 = _prox1 / num_neighbors
        prox2 = _prox2 / (num_neighbors * (num_neighbors - 1))
        return prox1, prox2
    

    def detect_one(self, v, es):
        adv_data = copy.deepcopy(self.data)
        adv_data.add_edges(to_undirected(torch.tensor(es).t()))

        p1, p2 = self._compute_score(self.surrogate, v, adv_data)
        if p1 > self.t1 or p2 > self.t2:
            return True
        else:
            return False 
            

    def detect_all(self):
        """ 1. trian a samplified GCN as a surrogate model
            2. get the embeddings of the nodes
            3. compute the threshold on unpertruabed edges
            4. compute the score for perturbed nodes
            5. check if it is perturbed.
        """
        surrogate = GCN(self.data.num_features, 16, self.data.num_classes, dropout=0.5, bias=False, activation=False).to(self.device)
        self._train_gcn(surrogate)
        
        challenge_edges = [tuple(e) for ee in self.e_star for e in ee]
        challenge_edges = list(set(challenge_edges))
        adv_data = copy.deepcopy(self.data)
        adv_data.add_edges(to_undirected(torch.tensor(challenge_edges).t()))

        # sample benign nodes
        # benign_nodes = random.sample(list(set(range(adv_data.num_nodes)) - set(self.v_star)), len(self.v_star))
        # labels = [1] * len(self.v_star) + [0] * len(benign_nodes)

        detection = []
        escaped_nodes = []
        t1, t2 = self._compute_threshold(surrogate)
        print('proximity threshold:', t1, t2)
        for v, es in tqdm(zip(self.v_star, self.e_star), total=len(self.v_star), desc='PD detecting'):
        # for v in tqdm(self.v_star + benign_nodes, desc='PD detecting'):
        # for es in tqdm(self.e_star, total=len(self.v_star), desc='PD detecting'):
            # u, v = es[0], es[1]
            # p1_u, p2_u = self._compute_score(surrogate, u, adv_data)
            p1_v, p2_v = self._compute_score(surrogate, v, adv_data)
            # print(f'Node {v}: Prox1 {p1_v:.4f}, Prox2 {p2_v:.4f}')
            if p1_v > t1 or p2_v > t2:
                detection.append(1)
            else:
                escaped_nodes.append(v)
                detection.append(0)
            # if (p1_u > t1 or p2_u > t2) and (p1_v > t1 or p2_v > t2):
            #     detection.append(1)
            # else:
            #     detection.append(0)
        return 0, np.sum(detection) / len(detection), escaped_nodes

        # print('Method: Jaccard Similarity')
        # print('number of nodes:', len(detection))
        # print('labels:', labels)
        # print('prediction:', detection)
        # print('confusion matrix:', confusion_matrix(labels, detection))
        # return f1_score(labels, detection), recall_score(labels, detection), precision_score(labels, detection)
    
    def discrepancy(self, data, posteriors, target_nodes=None):
        if target_nodes is not None:
            testing_nodes = target_nodes
        else:
            testing_nodes = data.test_set.nodes.tolist()
        # _, posteriors = model.predict(data, self.device, target_nodes=data.x.tolist(), return_posterior=True)
        prox1_list, prox2_list = [], []
        for v in testing_nodes:
            neighbors = list(data.adj_list[v])
            if len(neighbors) <= 1:
                continue
            
            _prox1, _prox2 = 0, 0
            for u in neighbors:
                _prox1 += self._kl_divergence(posteriors[v], posteriors[u])
                n_u = list(data.adj_list[u])
                for k in n_u:
                    _prox2 += self._kl_divergence(posteriors[u], posteriors[k])
            prox1 = _prox1 / len(neighbors)
            prox2 = _prox2 / (len(neighbors) * (len(neighbors) - 1))
            prox1_list.append(prox1)
            prox2_list.append(prox2)
            # print(f'Node {v}: Prox1 {prox1:.4f}, Prox2 {prox2:.4f}')
        return prox1_list, prox2_list
    

class GGD(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, activation=True, bias=True):
        super(GGD, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, bias=bias)
        self.gc2 = GraphConvolution(nhid, nclass, bias=bias)
        self.dropout = dropout
        self.activation = activation

        self.w_edge = nn.Linear(nclass, nclass, bias=False)
    
    def forward(self, u, v, x, adj):
        if self.activation:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return self.w_edge(x[u]).mm(x[v].t()).diag()
    
class GraphGenDetect(object):

    def __init__(self, data, v_star, e_star, device):
        self.data = data
        self.v_star = v_star
        self.e_star = e_star
        self.device = device

    def _train_generation_model(self, adv_data, subgraphs):

        model = GGD(self.data.num_features, 32, 16, dropout=0.5).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        model6_15 = []
        for e in tqdm(range(6), desc='Training GGD'):
            pi_list = []
            max_num_edges = 0
            for subset, edge_index in subgraphs:
                pi = torch.randperm(edge_index.size(1))
                pi_list.append(pi)
                max_num_edges = max(max_num_edges, edge_index.size(1))
            # print('max_num_edges:', max_num_edges)
            
            max_num_edges = min(max_num_edges, 10)
            for t in range(max_num_edges):
                losses = 0
                for i, (subset, edge_index) in enumerate(subgraphs):
                    if edge_index.size(1) <= t:
                        continue
                    
                    pi = pi_list[i]
                    adj = torch.zeros(subset.size(0), subset.size(0), device=self.device)
                    adj[edge_index[0, pi[:t]], edge_index[1, pi[:t]]] = 1
                    adj = adj + adj.t()
                    adj_tilde = torch.eye(adj.size(0)).to(self.device) + adj
                    adj_norm = utils.normalize(adj_tilde)

                    # Train a GCN to obtain embeddings
                    x = adv_data.x[subset].to(self.device)

                    node_pairs = []
                    labels = []
                    true_edges = list(map(tuple, edge_index[:, pi[t:]].t().tolist()))
                    for m in range(len(subset)):
                        for n in range(m + 1, len(subset)):
                            node_pairs.append((m, n))
                            if (m, n) in true_edges or (n, m) in true_edges:
                                labels.append(1)
                            else:
                                labels.append(0)
                    node_pairs = torch.tensor(node_pairs, device=self.device).t()
                    labels = torch.tensor(labels, device=self.device, dtype=torch.float32)

                    scores = model(node_pairs[0], node_pairs[1], x, adj_norm)
                    # loss = criterion(F.softmax(scores, dim=0), labels)
                    loss = criterion(torch.sigmoid(scores), labels)
                    # loss.backward()
                    # optimizer.step()
                    # optimizer.zero_grad()
                    losses += loss
                    # losses.append(loss.cpu().item())
                # l = torch.mean(losses)
                l = losses / len(subgraphs)
                l.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print(f'loss: {l.cpu().item()}')
            # if e >= 6:
            #     model6_15.append(copy.deepcopy(model))
        return model
                        
    def _check_edges(self, adv_data, subgraphs, model, edges_to_check):
        scores = defaultdict(dict)
        pi_list = []
        max_num_edges = 0
        for _, edge_index in subgraphs:
            pi_list.append(torch.randperm(edge_index.size(1)))
            max_num_edges = max(max_num_edges, edge_index.size(1)) 

        for t in range(max_num_edges):
            for i, (subset, edge_index) in enumerate(subgraphs):
                if edge_index.size(1) <= t:
                    continue
                node2idx = {v: i for i, v in enumerate(subset.tolist())}
                idx2node = {i: v for i, v in enumerate(subset.tolist())}
                pi = pi_list[i]

                containing_edges = []
                for u, v in edges_to_check:
                    if u not in node2idx or v not in node2idx:
                        continue
                    logits = torch.isin(edge_index[:, pi[t:]], torch.tensor([node2idx[u], node2idx[v]]))
                    if torch.sum(torch.logical_and(logits[0], logits[1])) > 0:
                        containing_edges.append((u, v))

                if len(containing_edges) == 0:
                    continue

                adj = torch.zeros(subset.size(0), subset.size(0), device=self.device)
                adj[edge_index[0, pi[:t]], edge_index[1, pi[:t]]] = 1
                adj = adj + adj.t()
                adj_tilde = torch.eye(adj.size(0)).to(self.device) + adj
                adj_norm = utils.normalize(adj_tilde)

                x = adv_data.x[subset].to(self.device)
                with torch.no_grad():
                    output = model(edge_index[0], edge_index[1], x, adj_norm)
                    # outputs = []
                    # for m in model_6_15:
                    #     m.eval()
                    #     output = m(edge_index[0], edge_index[1], x, adj_norm)
                    #     outputs.append(output)
                    # output = torch.stack(outputs, dim=0).mean(dim=0)
                for j, (u, v) in enumerate(edge_index[:, pi[t:]].t().tolist()):
                    _u, _v = idx2node[u], idx2node[v]
                    if (_u, _v) in containing_edges or (_v, _u) in containing_edges:
                        _e = tuple(sorted([_u, _v]))
                        if i not in scores[_e]:
                            scores[_e][i] = []
                        scores[(_e)][i].append(torch.sigmoid(output[pi[t:]][j]).cpu().item())
        result = []
        for u, v in edges_to_check:
            _e = tuple(sorted([u, v]))
            tmp = []
            for _, s in scores[_e].items():
                tmp.append(np.mean(s))
            result.append(statistics.harmonic_mean(tmp))
            # result.append(np.mean(tmp))
        return result

    def _sample_subgraphs(self, adv_data):
        subgraphs = []
        for v in range(adv_data.num_nodes):
            subset, edge_index, _, _ = k_hop_subgraph(v, 2, adv_data.edge_index, relabel_nodes=True)
            if edge_index.size(1) == 0:
                continue
            subgraphs.append((subset, utils.to_directed(edge_index)))
        return subgraphs


    def detect(self):
        """ 1. sample subgraphs (iterate each node and extract its 2-hop neighbors)
            2. randomly generate a permutation pi to get an edge sequence for each subgraph
            3. train a generaion model
                (1). 
                (2). feed the adj and feature to caculate the score for each node pair.
                (3). 
        """

        for challenge_edges in tqdm(self.e_star, desc='GGD detecting'):
            adv_data = copy.deepcopy(self.data)
            adv_data.add_edges(to_undirected(torch.tensor(challenge_edges).t()))

            subgraphs = self._sample_subgraphs(adv_data)
            model = self._train_generation_model(adv_data, subgraphs)
            scores = self._check_edges(adv_data, subgraphs, model, challenge_edges)
            print(scores)
    
    def detect_all(self):
        challenge_edges = [tuple(e) for ee in self.e_star for e in ee if e[0] != e[1]]
        challenge_edges = list(set(challenge_edges))

        adv_data = copy.deepcopy(self.data)
        adv_data.add_edges(to_undirected(torch.tensor(challenge_edges).t()))

        all_edges = [tuple(e) for e in adv_data.edge_index.t().tolist()]

        subgraphs = self._sample_subgraphs(adv_data)
        model = self._train_generation_model(adv_data, subgraphs)
        challenge_edges_scores = self._check_edges(adv_data, subgraphs, model, challenge_edges)
        print('Method: Graph Generation Detection')
        print('number of edges:', len(challenge_edges))
        print('scores:', challenge_edges_scores)

        scores = self._check_edges(adv_data, subgraphs, model, all_edges)
        sorted_indices = np.argsort(scores)[:len(challenge_edges)]
        top_k_edges = adv_data.edge_index.t()[sorted_indices].tolist()
        detection = []
        for e in challenge_edges:
            if e in top_k_edges or (e[1], e[0]) in top_k_edges:
                detection.append(1)
            else:
                detection.append(0)

        # print('scores', scores)
        # return np.sum(torch.sigmoid(torch.tensor(scores)).numpy() < 0.5) / len(scores)
        # return np.sum(np.array(scores) < 0.2) / len(scores)
        return np.sum(detection) / len(detection)
    

class JaccardSimilarity(object):

    def __init__(self, data, v_star, e_star, device):
        self.data = data
        self.v_star = v_star
        self.e_star = e_star
        self.device = device

    def detect_one(self, es):
        js = []
        for u, v in es:
            js.append(self._jaccard_similarity(self.data.x[u].numpy(), self.data.x[v].numpy()))
        # print('!!!!! js:', js)
        return np.sum(np.array(js) == 0) > len(es) * 0.5
    
    def detect(self, v, es):
        challenge_edges = [tuple(e) for e in es if e[0] != e[1]]
        challenge_edges = list(set(challenge_edges))

        adv_data = copy.deepcopy(self.data)
        adv_data.add_edges(to_undirected(torch.tensor(challenge_edges).t()))

        jaccard_sim = []
        for e in challenge_edges:
            u = self.data.x[e[0]].numpy()
            v = self.data.x[e[1]].numpy()
            jaccard_sim.append(self._jaccard_similarity(u, v))
        top_k_edges = np.array(challenge_edges)[np.array(jaccard_sim) == 0]

        top_k_edges_list = list(map(tuple, top_k_edges.tolist()))
        detection = []
        for e in challenge_edges:
            if e in top_k_edges_list or (e[1], e[0]) in top_k_edges_list:
                detection.append(1)
            else:
                detection.append(0)
        ratio_edge = np.sum(detection) / len(detection)
        nodes, counts = np.unique(top_k_edges.flatten(), return_counts=True)
        malicious_nodes = nodes[counts >= 5]
        if v in malicious_nodes:
            return True
        else:
            return False

    def detect_all(self):
        challenge_edges = [tuple(e) for ee in self.e_star for e in ee if e[0] != e[1]]
        challenge_edges = list(set(challenge_edges))

        adv_data = copy.deepcopy(self.data)
        adv_data.add_edges(to_undirected(torch.tensor(challenge_edges).t()))

        # All edges
        # edges = [tuple(e) for e in utils.to_directed(adv_data.edge_index).t().tolist()]
        # jaccard_sim = []
        # for e in edges:
        #     u = self.data.x[e[0]].numpy()
        #     v = self.data.x[e[1]].numpy()
        #     jaccard_sim.append(self._jaccard_similarity(u, v))

        # print('Method: Jaccard Similarity')
        # print('number of edges:', len(challenge_edges))
        # print('number of zeros:', (np.array(jaccard_sim) == 0).sum())
        # # print('jaccard similarity:', jaccard_sim)
        # sorted_indices = np.argsort(jaccard_sim)[:len(challenge_edges)]
        # top_k_edges = self.data.edge_index.t().numpy()[sorted_indices]
        
        edges = [tuple(e) for e in utils.to_directed(adv_data.edge_index).t().tolist()]
        jaccard_sim = []
        for e in edges:
            u = self.data.x[e[0]].numpy()
            v = self.data.x[e[1]].numpy()
            jaccard_sim.append(self._jaccard_similarity(u, v))
        top_k_edges = np.array(edges)[np.array(jaccard_sim) == 0]

        top_k_edges_list = list(map(tuple, top_k_edges.tolist()))
        detection = []
        for e in challenge_edges:
            if e in top_k_edges_list or (e[1], e[0]) in top_k_edges_list:
                detection.append(1)
            else:
                detection.append(0)
        print('number of edges that are detected:', np.sum(detection))
        ratio_edge = np.sum(detection) / len(detection)
        return ratio_edge >= 0.5
        nodes, counts = np.unique(top_k_edges.flatten(), return_counts=True)
        malicious_nodes = nodes[counts > 0.5 * len(challenge_edges)]

        detection = []
        escaped_nodes = list(set(self.v_star) - set(malicious_nodes.tolist()))
        for n in malicious_nodes:
            if n in self.v_star:
                detection.append(1)
            else:
                detection.append(0)
        ratio_node = np.sum(detection) / len(self.v_star)

        return ratio_edge, ratio_node, escaped_nodes
        # nodes, counts = np.unique(top_k_edges.flatten(), return_counts=True)
        # malicious_nodes = nodes[counts >= 5]

        # detection = []
        # for n in malicious_nodes:
        #     if n in self.v_star:
        #         detection.append(1)
        #     else:
        #         detection.append(0)

        # return np.sum(detection) / len(self.v_star)
        # return (np.array(jaccard_sim) == 0).sum() / len(jaccard_sim)


        # random sample the edges with the number of challenge edges
        # all_edges = [tuple(e) for e in self.data.edge_index.t().tolist()]
        # benign_edges = random.sample(all_edges, len(challenge_edges))
        # edges = challenge_edges + benign_edges
        # labels = [1] * len(challenge_edges) + [0] * len(benign_edges)

        # for e in edges:
        #     u = self.data.x[e[0]].numpy()
        #     v = self.data.x[e[1]].numpy()
        #     jaccard_sim.append(self._jaccard_similarity(u, v))
        
        # preds = np.where(np.array(jaccard_sim) == 0, 1, 0)

        # print('Method: Jaccard Similarity')
        # print('number of edges:', edges)
        # print('labels:', labels)
        # print('prediction:', preds.tolist())
        # print('confusion matrix:', confusion_matrix(labels, preds))

        # # F1 score
        # f1 = f1_score(labels, preds)
        # return f1, recall_score(labels, preds), precision_score(labels, preds)

    def _jaccard_similarity(self, u, v):
        """
        J_u,v = M_11 / (M_01 + M_10 + M_11)
        """
        M_11 = np.sum(u * v)
        M_01 = np.sum(u * (1 - v))
        M_10 = np.sum((1 - u) * v)
        return M_11 / (M_01 + M_10 + M_11)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--detector', type=str, default='linkpred')
    parser.add_argument('--num_nodes', type=int, default=50)
    parser.add_argument('--num-perts', type=int, default=10)
    parser.add_argument('--num-trials', type=int, default=5)
    parser.add_argument('--attack', type=str, default='ours')
    parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=1)
    args = parser.parse_args()

    ts = int(time.time())
    device = utils.get_device(args) 
    result = defaultdict(list)
    for t in range(args.num_trials):
    # for t in [2, 3, 4]:
        for method in tqdm(['ig', 'sga', 'fga', 'rnd'], desc=f'At trial {t}'):
        # for method in tqdm(['ours'], desc=f'At trial {t}'):
            data_filename = os.path.join('archive', str(t), args.dataset, f'data_{method}.pkl')
            with open(data_filename, 'rb') as f:
                data = pickle.load(f)
            v_star_filename = os.path.join('archive', str(t), args.dataset, f'v_star_{method}.pkl')
            with open(v_star_filename, 'rb') as f:
                v_star = pickle.load(f)
            e_star_filename = os.path.join('archive', str(t), args.dataset, f'e_star_{method}.pkl')
            with open(e_star_filename, 'rb') as f:
                e_star = pickle.load(f)

            # detector = LinkPredDetector(data, v_star, e_star, device)
            # r = detector.detect()
            # result['trial'].append(t)
            # result['method'].append(method)
            # result['detector'].append('linkpred')
            # result['detection ratio'].append(r)

            detector = OutlierDetector(data, v_star, e_star)
            r = detector.detect()
            result['trial'].append(t)
            result['method'].append(method)
            result['detector'].append('outlier')
            result['detection ratio'].append(r)

            detector = ProximityDetector(data, v_star, e_star, device)
            r = detector.detect()
            result['trial'].append(t)
            result['method'].append(method)
            result['detector'].append('pd')
            result['detection ratio'].append(r)

        _df = pd.DataFrame(result)
        _df = _df[_df['trial'] == t]
        print('trial:', t)
        print(_df.groupby(['method', 'detector']).mean()['detection ratio'])
        print('-' * 80)
    
    df = pd.DataFrame(result)
    df.groupby(['method', 'detector']).mean()['detection ratio']
    df.to_csv(os.path.join('result', 'detection{ts}.csv'), index=False)
