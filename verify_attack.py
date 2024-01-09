"""
Minimum Min-Max attack

Kun Wu
Stevens Institute of Technology
"""


import time
import math
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import utils

class Minimum_MinMax_Attack:

    def __init__(self, model, data, delta=None, budget=2, epochs=200, lr=0.001, alpha=1, beta=0.05, device='cpu') -> None:
        self.model = model
        self.data = data
        self.ori_adj = data.adjacency_matrix()
        self.epochs = epochs

        self.lr = lr
        self.alpha = alpha
        self._alpha = self.alpha / 100
        self.beta = beta
        self._beta = 0.001

        self.device = device
        
        self.epsilon = 1e-4

        # the pertubation vector
        if delta is None:
            self.delta = nn.Parameter(torch.zeros(int(self.data.num_nodes * (self.data.num_nodes - 1) / 2), device=self.device))
            self.delta.data.fill_(0.0001)
        else:
            self.delta = delta

        # budget
        self.budget = budget

    def _minmax_attack(self, victim_model, features, ori_adj, labels, idx_train, alpha):
        optimizer = optim.Adam(victim_model.parameters(), lr=self.lr)

        victim_model.eval()
        for t in range(100):
            victim_model.train()
            modified_adj = self._get_modified_adj(ori_adj, self.delta)
            adj_norm = utils.normalize(modified_adj)
            output = victim_model(features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])
            # outputs = victim_model(features, adj_norm)[self.data.train_set.nodes]
            # loss = self.ce_loss(outputs, self.data.train_set.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # PGD attack
            victim_model.eval()
            modified_adj = self._get_modified_adj(ori_adj, self.delta)
            adj_norm = utils.normalize(modified_adj)
            output = victim_model(features, adj_norm)
            loss = self._loss(output[idx_train], labels[idx_train])
            adj_grad = torch.autograd.grad(loss, [self.delta])[0]
            adj_grad = torch.nan_to_num(adj_grad) 

            lr = alpha / (t + 1)
            self.delta.data.add_(lr * adj_grad)
        
            # _delta = copy.deepcopy(self.delta.data.detach())
            # _delta += alpha * adj_grad
            self.projection()
            # print('     CW loss:', loss)
            # if loss > 0:
            #     break

        # _delta = self.random_sample(ori_adj, features, _delta, v, label)
        # _delta = self._projection(_delta)
        self.random_sample(ori_adj, features, labels, idx_train)
        # return _delta
    
    def _evaluate(self, victim_model, features, ori_adj, v, label):
        victim_model.eval()
        with torch.no_grad():
            _adj = self._get_modified_adj(ori_adj, self.delta, v)
            _adj_norm = utils.normalize(_adj)
            output = victim_model(features, _adj_norm)[v]
            cw_loss = self._loss(output, label)
        return cw_loss
    
    def attack(self, features, adj, labels, idx_train):
        victim_model = self.model.model.to(self.device)
        features = features.to(self.device)
        ori_adj = adj.to_dense().to(self.device)
        labels = labels.to(self.device)

        # # reset mask
        # self.mask[:] = 1

        # best_delta = None
        # candidates = []
        victim_model.eval()
        self._minmax_attack(victim_model, features, ori_adj, labels, idx_train, self.alpha)
        modified_adj = self._get_modified_adj(ori_adj, self.delta)
        perturbed_edges = np.array(np.where(modified_adj != ori_adj))
        perturbed_edges = perturbed_edges[:, perturbed_edges[0] < perturbed_edges[1]]
        E_t = list(map(tuple, perturbed_edges.T.tolist()))
        # print(E_t)
        return E_t

        # for t in tqdm(range(self.epochs), desc='verify attack'):
        # for t in range(5):
        #     alpha = self.alpha

        #     cw_loss = self._evaluate(victim_model, features, ori_adj, v, label)
        #     if cw_loss > 0:
        #         u = self.delta.data.nonzero().item()
        #         if u not in candidates:
        #             candidates.append(u)
        #             # self.mask[u] = 0
        #     else:
        #         break
        # if len(candidates) == 0:
        #     return None
        # return [(v, u) for u in candidates]

    # def attack(self, features, adj, label, v):
    #     victim_model = self.model.model.to(self.device)
    #     features = features.to(self.device)
    #     ori_adj = adj.to_dense().to(self.device)
    #     label = label.to(self.device)

    #     tried_budget = {}
    #     max_budget = 10
    #     min_budget = 2

    #     best_delta = None
    #     victim_model.eval()
    #     # for t in tqdm(range(self.epochs), desc='verify attack'):
    #     with tqdm(total=self.epochs, desc='verify attack') as pbar:
    #         for t in range(self.epochs):
    #             # print('-' * 20, t, '-' * 20)
    #             alpha = self.alpha
    #             beta = self.beta

    #             self._minmax_attack(victim_model, features, ori_adj, v, label, alpha)
    #             cw_loss = self._evaluate(victim_model, features, ori_adj, v, label)            
    #             is_minimum = False
    #             if cw_loss > 0:
    #                 tried_budget[self.budget] = True
    #                 if best_delta is None or self.delta.data.norm(p=0) < best_delta.norm(p=0):
    #                     best_delta = copy.deepcopy(self.delta.data.detach())

    #                 for b in range(min_budget, self.budget+1):
    #                     if b not in tried_budget:
    #                         break
    #                     is_minimum = True
    #                 self.budget = max(min_budget, min(self.budget - 1, (1 - beta) * self.budget))
    #                 while self.budget in tried_budget and self.budget > min_budget:
    #                     self.budget = max(min_budget, min(self.budget - 1, (1 - beta) * self.budget))
    #             else:
    #                 tried_budget[self.budget] = False
    #                 self.budget = max(self.budget + 1, (1 + beta) * self.budget)
    #                 while self.budget in tried_budget:
    #                     self.budget = max(self.budget + 1, (1 + beta) * self.budget)

    #             # print('-' * 50)
    #             # if best_delta is not None:
    #             #     print('  best delta norm:', best_delta.norm(p=0))
    #             # print('  new budget:', self.budget)
    #             # print('  current delta norm:', self.delta.norm(p=0))
    #             # print('  cw loss:', cw_loss)
    #             # print('  tried budget:', tried_budget)
    #             # print('-' * 50)

    #             if is_minimum:
    #                 # print('finished minimum attack.')
    #                 break
                
    #             if self.budget >= max_budget:
    #                 if best_delta is None:
    #                     print('Hit the maximum budget, did not find best delta, return None.')
    #                 else:
    #                     print('Hit the maximum budget, the norm of the best delta:', best_delta.norm(p=0))
    #                 break
    #             pbar.update(1)
    #             pbar.set_description(f'verify attack (budget: {self.budget})')
 
    #     if best_delta is None:
    #         return None
    #     _nodes = torch.nonzero(best_delta).squeeze().tolist()
    #     if isinstance(_nodes, int):
    #         E_t = [(v, _nodes)]
    #     else:
    #         E_t = [(v, u) for u in _nodes]
        
    #     if len(E_t) == 0:
    #         print('!!!!!!!')
    #         print('1', self.delta.norm(p=0))
    #         print('2', best_delta.norm(p=0))
    #         print('3', self.budget)
    #         return None

    #     return E_t
    
    def _projection(self, delta):
        sorted_delta, _ = torch.sort(delta, descending=True)
        miu = sorted_delta[int(self.budget) + 1]
        _delta = delta - miu
        _delta = torch.where(_delta > 0, torch.ones_like(_delta), torch.zeros_like(_delta))
        return _delta

    def l0_projection_0(self, delta):
        delta_abs = delta.flatten().abs()
        sorted_indices = delta_abs.argsort(dim=0, descending=True).gather(1, self.budget)
        thresholds = delta_abs.gather(0, sorted_indices)
        return torch.clamp(torch.where((delta_abs >= thresholds).view_as(delta), delta, torch.zeros(1, device=self.device)), min=0, max=1)

    def projection(self):
        if torch.clamp(self.delta, 0, 1).sum() > self.budget:
            left = (self.delta - 1).min()
            right = self.delta.max()
            miu = self.bisection(left, right, self.budget, epsilon=self.epsilon)
            self.delta.data.copy_(torch.clamp(self.delta.data - miu, min=0, max=1))
            # else:
            #     delta = torch.clamp(delta.data - miu, min=0, max=1)
        else:
            self.delta.data.copy_(torch.clamp(self.delta.data, min=0, max=1))
            # if isinstance(delta, nn.Parameter):
            #     delta.data.copy_(torch.clamp(delta, min=0, max=1))
            # else:
            #     delta = torch.clamp(delta, min=0, max=1)
        # return delta
    
    def random_sample(self, ori_adj, features, labels, idx_train):
        K = 20
        best_loss = -1000
        best_s = None
        victim_model = self.model.model
        victim_model.eval()
        with torch.no_grad():
            s = self.delta.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)

                # print(sampled.sum())
                if sampled.sum() > self.budget:
                    continue
                # self.delta.data.copy_(torch.tensor(sampled))
                modified_adj = self._get_modified_adj(ori_adj, self.delta)
                modified_adj_norm = utils.normalize(modified_adj)
                output = victim_model(features, modified_adj_norm)
                loss = self._loss(output[idx_train], labels[idx_train])
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled

            if best_s is not None:
                self.delta.data.copy_(torch.tensor(best_s))
            else:
                print('WARNING: cannot find best s during random sample, the budget is', self.budget)

        # if best_s is None:
        #     # print('loss:', loss)
        #     print('sampled sum:', sampled.sum())
        #     print('delta:', torch.min(delta), torch.mean(delta), torch.max(delta))
        #     raise RuntimeError(f'Did not find s during random sample.')
        # else:
        #     return torch.from_numpy(best_s).float().to(self.device)

    def ce_loss(self, outputs, labels):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        return loss

    def _loss(self, output, labels):
        """
        Carlili-Wagner Loss
        """
        # onehot = torch.zeros(self.data.num_classes, device=self.device)
        eye = torch.eye(self.data.num_classes, device=self.device)
        onehot = eye[labels]
        best_second_class = (output - 1000 * onehot).argmax(1)
        # loss = output[label] - output[best_second_class]
        # reverse the order so that can run minimization instead of maximization
        margin = output[np.arange(len(output)), labels] - \
                   output[np.arange(len(output)), best_second_class]
        # loss = output[best_second_class] - output[label]
        k = 0
        loss = -torch.clamp(margin, min=k).mean()
        return loss


    def _get_modified_adj(self, ori_adj, delta):
        m = torch.zeros_like(ori_adj, device=self.device)
        tril_indices = torch.tril_indices(row=self.data.num_nodes, col=self.data.num_nodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = delta
        m = m + m.t()

        modified_adj = copy.deepcopy(ori_adj)
        modified_adj = modified_adj + (1 - modified_adj) * m
        # pert = ori_adj[v] + (1 - ori_adj[v]) * m
        # modified_adj[v] = pert
        # modified_adj[:, v] = pert

        return modified_adj
    
    def bisection(self, a, b, budget, epsilon=1e-5):
        def func(x):
            return torch.clamp(self.delta-x, 0, 1).sum() - budget
        
        # print('a', a, ', b', b, ', budget:', budget)

        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu


# if __name__ == '__main__':
#     import argument
#     import data_loader

#     parser = argument.load_parser()
#     args = parser.parse_args()

#     data = data_loader.load(args)

#     for _ in tqdm(range(100), desc='attack'):
#         v = random.choice(list(range(data.num_nodes)))
#         print(f'attacking node {v}, degree: {data.degree(v)}')
#         time.sleep(1)

#     time.sleep(3)
#     print(f'The ASR: {1.0}, and the incompleteness probability: {0.74}')
