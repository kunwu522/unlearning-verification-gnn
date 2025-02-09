import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.utils import k_hop_subgraph, to_undirected
from sklearn.metrics import classification_report

import utils
from model.gcn import GCN
from .unlearn import Unlearn


def propagate(features, k, adj_norm):
    feature_list = []
    feature_list.append(features)
    for i in range(k):
        feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
    return feature_list[-1]

def criterionKD(p, q, T=1.5):
    loss_kl = nn.KLDivLoss(reduction="batchmean")
    soft_p = F.log_softmax(p / T, dim=1)
    soft_q = F.softmax(q / T, dim=1).detach()
    return loss_kl(soft_p, soft_q)


class GATE(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lr = torch.nn.Linear(dim, dim)

    def forward(self, x):
        t = x.clone()
        return self.lr(t)


class MEGU(Unlearn):
    """ An implementation of the MEGU algorithm for unlearning graph neural networks.
        The algorithm is described in the paper 
        "Towards Effective and General Graph Unlearning via Mutual Evolution"
        by
        Xunkai Li, Yulin Zhao, Zhengyu Wu, Wentao Zhang, Rong-Hua Li, Guoren Wang

        Based on the implementation from
        https://github.com/xkLi-Allen/MEGU
    """

    def __init__(
            self, 
            seed, features, adj, labels, config, device, 
            model_type='gcn', epochs=1000, verbose=False,
            unlearn_lr=0.09, kappa=0.01, patience=10,
            edge_index=None, alpha1=0.48, alpha2=0.24
        ):
        super(MEGU, self).__init__(seed, features, adj, labels, config, device, model_type, epochs, verbose)
        self.name = 'MEGU'

        self.unlearn_lr = unlearn_lr
        self.kappa = kappa
        self.num_classes = self.labels.max().item() + 1
        self.num_layers = 2
        self.patience = patience
        self.edge_index = edge_index
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.model = GCN(
            nfeat=self.features.shape[1],
            nhid=self.config['nhid'],
            nclass=self.num_classes,
            dropout=self.config['dropout']
        ).to(self.device)

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


    def unlearn(self, edges_to_unlearn):
        self.retrain_model = copy.deepcopy(self.model)
        self.adj_prime = copy.deepcopy(self.adj)

        # remove edges from the adjacency matrix
        _edge_index = to_undirected(torch.tensor(edges_to_unlearn).t())
        self.adj_prime[_edge_index[0], _edge_index[1]] = 0
        self.adj_prime_norm = utils.normalize(self.adj_prime + torch.eye(self.adj_prime.shape[0], device=self.device))
        # self.adj_prime_norm = self.adj_prime_norm.to(self.device)

        self.edge_index_unlearn = torch.cat(torch.where(self.adj_prime != 0)).view(2, -1)

        self.temp_node = np.unique([v for e in edges_to_unlearn for v in e])
        # print('Temp node:', self.temp_node)
        self.neighbor_khop = self.neighbor_select(self.features)

        self.operator = GATE(self.num_classes).to(self.device)
        optimizer = torch.optim.SGD([
            {'params': self.retrain_model.parameters()},
            {'params': self.operator.parameters()}
        ], lr=0.001)
        # optimizer = torch.optim.Adam([
        #     {'params': self.retrain_model.parameters()},
        #     {'params': self.operator.parameters()}
        # ], lr=0.001)

        with torch.no_grad():
            self.model.eval()
            preds = self.retrain_model(self.features, self.adj_norm)
            self.preds = torch.argmax(preds, dim=1)

        for epoch in range(30):
            self.retrain_model.train()
            self.operator.train()
            optimizer.zero_grad()

            out_ori = self.retrain_model(self.features, self.adj_prime_norm)
            out = self.operator(out_ori)

            loss_u = criterionKD(out_ori[self.temp_node], out[self.temp_node]) - F.cross_entropy(out[self.temp_node], self.preds[self.temp_node])
            # loss_u = -F.cross_entropy(out[self.temp_node], preds[self.temp_node])
            loss_r = criterionKD(out[self.neighbor_khop], out_ori[self.neighbor_khop]) + F.cross_entropy(out_ori[self.neighbor_khop], self.preds[self.neighbor_khop])
            loss = self.kappa * loss_u + loss_r

            loss.backward()
            optimizer.step()

        self.retrain_model.eval()
        with torch.no_grad():
            test_out = self.retrain_model(self.features, self.adj_prime_norm)
            # test_out = self.operator(test_out)
        y_preds = torch.argmax(self.correct_and_smooth(F.softmax(test_out, dim=-1), self.preds)[self.idx_test], dim=1)
        y_true = self.labels[self.idx_test].cpu().numpy()
        result = classification_report(y_true, y_preds.cpu().numpy(), output_dict=True, zero_division=0)

        return result
    
    def predict(self, target_nodes, use_retrained=False, return_posterior=False):
        model = self.retrain_model if use_retrained else self.model
        adj_norm = self.adj_prime_norm if use_retrained else self.adj_norm

        model.eval()
        with torch.no_grad():
            outputs = model(self.features, adj_norm)
            if use_retrained:
                self.operator.eval()
                _outputs = self.operator(outputs)

        if use_retrained:
            _post = self.correct_and_smooth(F.softmax(outputs, dim=-1), self.preds).cpu().numpy()[target_nodes]
            post = self.correct_and_smooth(F.softmax(_outputs, dim=-1), self.preds).cpu().numpy()[target_nodes]
            y_pred = np.argmax(post, axis=1)
            _y_pred = np.argmax(_post, axis=1)
            print(
                f'target nodes: {target_nodes}', 
                f'y_pred: {y_pred}, ori preds: {torch.argmax(outputs[target_nodes], dim=1).cpu().numpy()}',
                f'P_star: {np.argmax(_post, axis=1)}, ori P_star: {torch.argmax(_outputs[target_nodes], dim=1).cpu().numpy()}')  
        else:
            post = F.softmax(outputs[target_nodes], dim=-1).cpu().numpy()
            y_pred = torch.argmax(outputs[target_nodes], dim=1).cpu().numpy()

        # y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        if return_posterior:
            # post = F.softmax(outputs[target_nodes], dim=-1).cpu().numpy()
            return y_pred, post
        else:
            return y_pred
        
    def parameters(self, use_retrained=False):
        model = self.retrain_model if use_retrained else self.model
        return model.parameters()

    def neighbor_select(self, features):
        temp_features = features.clone()
        pfeatures = propagate(temp_features, self.num_layers, self.adj_norm)
        reverse_feature = self.reverse_features(temp_features)
        re_pfeatures = propagate(reverse_feature, self.num_layers, self.adj_norm)

        cos = nn.CosineSimilarity()
        sim = cos(pfeatures, re_pfeatures)
        
        alpha = 0.1
        gamma = 0.1
        max_val = 0.
        while True:
            influence_nodes_with_unlearning_nodes = torch.nonzero(sim <= alpha).flatten().cpu()
            if len(influence_nodes_with_unlearning_nodes.view(-1)) > 0:
                temp_max = torch.max(sim[influence_nodes_with_unlearning_nodes])
            else:
                alpha = alpha + gamma
                continue

            if temp_max == max_val:
                break

            max_val = temp_max
            alpha = alpha + gamma

        # influence_nodes_with_unlearning_nodes = torch.nonzero(sim < 0.5).squeeze().cpu()
        neighborkhop, _, _, two_hop_mask = k_hop_subgraph(
            torch.tensor(self.temp_node),
            self.num_layers,
            self.edge_index,
            num_nodes=self.features.shape[0])

        neighborkhop = neighborkhop[~np.isin(neighborkhop.cpu(), self.temp_node)]
        neighbor_nodes = []
        for idx in influence_nodes_with_unlearning_nodes:
            # if idx in neighborkhop and idx not in self.temp_node:
            if idx not in self.temp_node:
                neighbor_nodes.append(idx.item())

        # print('HIN:', neighbor_nodes)        
        neighbor_nodes_mask = torch.from_numpy(np.isin(np.arange(self.features.shape[0]), neighbor_nodes))

        return neighbor_nodes_mask

    def reverse_features(self, features):
        reverse_features = features.clone()
        for idx in self.temp_node:
            reverse_features[idx] = 1 - reverse_features[idx]

        return reverse_features
    
    def correct_and_smooth(self, y_soft, preds):
        pos = CorrectAndSmooth(num_correction_layers=80, correction_alpha=self.alpha1,
                               num_smoothing_layers=80, smoothing_alpha=self.alpha2,
                               autoscale=False, scale=0.2)
        train_mask = torch.zeros(self.features.shape[0], dtype=torch.bool)
        train_mask[self.idx_train] = 1
        y_soft = pos.correct(
            y_soft, preds[train_mask], train_mask, self.edge_index_unlearn
        )
        y_soft = pos.smooth(
            y_soft, preds[train_mask], train_mask, self.edge_index_unlearn
        )

        # print('y soft:', y_soft)
        # if y_soft.shape[1] > 1:
        #     return y_soft
        # else:
        # return torch.argmax(y_soft, dim=1)
        return y_soft