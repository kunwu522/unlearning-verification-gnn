import math
import time
from tqdm import tqdm
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

import utils
from .preprocess import preprocess


class MiBTack(object):

    def __init__(self, model, data, device=torch.device('cpu'), **kwargs) -> None:
        self.model = model
        self.data = data
        self.device = device

        self.target = kwargs['target'] if 'target' in kwargs else 'gcn'
        self.dataset = kwargs['dataset'] if 'dataset' in kwargs else 'cora'
        self.tanh = kwargs['tanh'] if 'tanh' in kwargs else 1.0
        self.explore = kwargs['explore'] if 'explore' in kwargs else True
        self.decay = kwargs['decay'] if 'decay' in kwargs else True
        self.gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.
        self.norm = kwargs['norm'] if 'norm' in kwargs else 0
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 1
        self.verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        self.batch_view = lambda tensor: tensor.view(self.batch_size, *[1] * (self.data.num_nodes - 1))

    def attack(self, v, label, prediction):
        if self.explore:
            init_state = preprocess(self.target, self.dataset, self.model, self.data, v, prediction, self.device)

        self.model.to(self.device)
        self.model.eval()

        bound, acc_list, ptb_list = [], [], []
        best_delta_dict = {}
        margin = {}

        if self.verbose:
            print(f'Start attack node {v}...')


        adv_nodes = []
        for _ in range(5):
            alpha_init = 1.0
            alpha_final = alpha_init / 100
            gamma_init = 0.05
            gamma_final = 0.001

            labels = prediction

            adj_changes_org = nn.Parameter(torch.empty(self.data.num_nodes, device=self.device, dtype=torch.float32)).unsqueeze(0)
            adj_changes_org[0].data.fill_(0.0001)
            if self.explore:
                with torch.no_grad():
                    adj_changes_org[0, init_state[v]] = 0.001
            adj_changes = adj_changes_org * adj_changes_org
            complementary = None
            epsilon = torch.ones(self.batch_size, device=self.device)
            worst_norm = torch.max(adj_changes, 1 - adj_changes).norm(p=self.norm, dim=1)
            best_norm = worst_norm.clone()

            best_delta = None
            adv_found = False

            ori_features = self.data.x.to(self.device)
            adj = self.data.adjacency_matrix().to_dense().to(self.device)

            # print('prediction', prediction, ', label', label)
            logits = F.log_softmax(self.model(ori_features, utils.normalize(adj)), dim=1)[v] # Following the original code of MiBTack
            p_y = math.exp(logits[labels].item())
            logits[labels] = -1000
            pred_max_c = logits.argmax()
            p_c = math.exp(logits[pred_max_c].item())
            if p_c - p_y > self.gamma:
                best_norm = 0
                bound.append(0)
                acc_list.append(True)
                ptb_list.append(None)
                best_delta_dict[v] = None
                margin[v] = p_c - p_y
                if self.verbose:
                    print("111, check:  target:", v, f"p_y {labels}:", p_y, f', p_c {pred_max_c}:', p_c)
                return None
            
            if self.explore:
                index_adv = torch.tensor([[v, init_state[v]], [init_state[v], v]], dtype=torch.long, device=self.device)
                values_adv = torch.tensor([-2 * adj[v, init_state[v]] + 1, -2 * adj[init_state[v], v] + 1], dtype=torch.float32, device=self.device)
                adj_tmp = torch.sparse.FloatTensor(index_adv, values_adv, torch.Size([self.data.num_nodes, self.data.num_nodes])).to(self.device)
                logits = F.log_softmax(self.model(ori_features, utils.normalize((adj + adj_tmp).to_dense())), dim=1)[v]
                p_y = math.exp(logits[labels].item())
                logits[labels] = -1000
                pred_max_c = logits.argmax()
                p_c = math.exp(logits[pred_max_c].item())
                if p_c - p_y > self.gamma:
                    best_norm = 1.0
                    bound.append(1.0)
                    acc_list.append(True)
                    ptb_list.append((v, init_state[v]))
                    best_delta = adj_tmp
                    best_delta_dict[v] = best_delta
                    margin[v] = p_c - p_y
                    if self.verbose:
                        print("222, check:  target:", v, f"p_y {labels}:", p_y, f', p_c {pred_max_c}:', p_c)
                    return None
                
            t = 0
            epoch = 800
            patience = 800
            L0_list = []
            while True:
                if adv_found:
                    patience -= 1
                if patience == 0:
                    break

                alpha = alpha_init
                gamma = gamma_init
                if adv_found and self.decay:
                    cosine = (1 + math.cos(math.pi * (epoch - patience) / epoch)) / 2
                    alpha = alpha_final + (alpha_init - alpha_final) * cosine
                    gamma = gamma_final + (gamma_init - gamma_final) * cosine
                
                delta_norm = adj_changes.data.norm(p=self.norm, dim=1)
                L0_list.append(delta_norm)
                modified_adj = self._get_modified_adj(complementary, adj_changes, adj, v, self.device, adv_nodes)

                adj_norm = utils.normalize(modified_adj)
                logits = F.log_softmax(self.model(ori_features, adj_norm), dim=1)[v].unsqueeze(0)
                pred_labels = logits.argmax(dim=1)
                if t == 0:
                    labels_infhot = torch.zeros_like(logits.detach(), dtype=torch.float).scatter(1, torch.tensor(labels, device=self.device).view(1, 1), float('inf'))
                    logit_diff_func = partial(self._difference_of_logits, labels=labels, labels_infhot=labels_infhot)
                loss = -logit_diff_func(logits=logits)
                delta_grad = grad(loss.sum(), adj_changes_org, only_inputs=True, retain_graph=True)[0]

                if t > 0:
                    cols = adj_changes.data.nonzero()[:, 1].tolist()
                    rows = [v] * len(cols)
                    ori_values = (-2 * adj[v].to_dense()[cols].detach().cpu().numpy() + 1).tolist()
                    index_adv = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
                    values_adv = torch.tensor(ori_values, dtype=torch.float32, device=self.device)
                    adj_tmp = torch.sparse.FloatTensor(index_adv, values_adv, torch.Size([self.data.num_nodes, self.data.num_nodes])).to(self.device)
                    adj_tmp = adj_tmp + adj_tmp.t()

                    logits_hard = F.log_softmax(self.model(ori_features, utils.normalize((adj + adj_tmp).to_dense())), dim=1)[v]
                    p_y = math.exp(logits_hard[labels].item())
                    logits_hard[labels] = -1000
                    pred_max_c_hard = logits_hard.argmax()
                    p_c = math.exp(logits_hard[pred_max_c_hard].item())
                    is_adv = torch.tensor([p_c - p_y > self.gamma], dtype=torch.bool, device=self.device)

                    is_smaller = delta_norm < best_norm
                    is_both = is_adv & is_smaller
                    adv_found = (adv_found + is_adv) > 0
                    best_norm = torch.where(is_both, delta_norm, best_norm)
                    best_delta = adj_tmp.data if is_both[0] else best_delta
                    if is_both[0]:
                        margin[v] = p_c - p_y
                    if delta_norm < 2 and is_adv:
                        if self.verbose:
                            print("early stop since δ_norm:", delta_norm, " is_adv:", is_adv)
                        break
                    if self.norm == 0:
                        epsilon = torch.where(is_adv, 
                                            torch.min(torch.min(epsilon - 1, (epsilon * (1 - gamma)).float()), best_norm), 
                                            torch.max(epsilon + 1, (epsilon * (1 + gamma)).float()))
                        epsilon.clamp_min_(0)
                    else:
                        distance_to_boundary = loss.detach().abs() / delta_grad.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
                        epsilon = torch.where(is_adv,
                                            torch.min(epsilon * (1 - gamma), best_norm),
                                            torch.where(adv_found, epsilon * (1 + gamma), delta_norm + distance_to_boundary))

                    epsilon = torch.min(epsilon, worst_norm)

                grad_l2_norms = delta_grad.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
                delta_grad.div_(grad_l2_norms.view(self.batch_size, 1))

                adj_changes_org.data.add_(delta_grad, alpha=alpha)
                adj_changes = torch.tanh(self.tanh * adj_changes_org * adj_changes_org)

                # project
                adj_changes.data = self._l0_projection_0(adj_changes, epsilon)
                t += 1

            if self.verbose:
                print('best norm:', best_norm, ', best delta:', best_delta)
            nodes = adj_changes.squeeze().nonzero().cpu().detach().squeeze().tolist()
            if isinstance(nodes, list):
                adv_nodes.extend(nodes)
                # E_t.extend([(v, u) for u in nodes])
            elif isinstance(nodes, int):
                adv_nodes.append(nodes)
                # E_t.append((v, nodes))
            else:
                raise ValueError(f'Invalid nodes: {nodes}')

        return [(v, u) for u in adv_nodes]

    def _get_modified_adj(self, complementary, adj_changes, adj, target_node, device, changed_nodes=None):
        if complementary is None:
            complementary = torch.ones_like(adj_changes[0], device=device)
            complementary = complementary - adj[target_node]
            complementary[target_node] = 0.0
            if changed_nodes is not None:
                complementary[changed_nodes] = 0.0
            # complementary = (complementary - adj[target_node]) - adj[target_node]
            complementary = complementary.unsqueeze(0)
        adj[target_node, :] = adj[target_node, :] + adj_changes * complementary
        adj[:, target_node] = adj[:, target_node] + torch.squeeze((adj_changes * complementary).t())
        return adj

    def _difference_of_logits(self, logits: torch.Tensor, labels: torch.Tensor,
                            labels_infhot: Optional[torch.Tensor] = None) -> torch.Tensor:
        if labels_infhot is None:
            labels_infhot = torch.zeros_like(logits.detach()).scatter(1, labels.unsqueeze(1), float('inf'))
        #     print(logits,labels,labels.unsqueeze(1))
        class_logits = logits.gather(1, torch.tensor(labels, device=self.device).view(1, 1)).squeeze(1)
        other_logits = (logits - labels_infhot).max(1).values
        return class_logits - other_logits

    def _l0_projection_0(self, delta: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        delta_abs = delta.flatten(1).abs()
        sorted_indices = delta_abs.argsort(dim=1, descending=True).gather(1, (epsilon.long().unsqueeze(1) - 1).clamp_min(0))
        thresholds = delta_abs.gather(1, sorted_indices)
        return torch.clamp(torch.where((delta_abs >= thresholds).view_as(delta), delta, torch.zeros(1, device=delta.device)), min=0, max=1)

        # sorted_indices = delta_abs.argsort(dim=1, descending=True)[:, :5]
        # projected = torch.zeros_like(delta)
        # projected[:, sorted_indices.squeeze()] = delta[:, sorted_indices.squeeze()]
        # print('!!!!!', projected.size(), projected.nonzero())
        # return projected



# def attack(args, surrogate, data, node_token, prediction, device):
#     if args.explore:
#         init_state = preprocess(args, surrogate, data, node_token, prediction, device)

#     batch_size = 1
#     norm = 0

#     tanh = 1.0
#     decay = True
#     gamma = 0.2

#     bound = []
#     acc_list = []
#     ptb_list = []
#     order = 0
#     epoch_list = []

#     t0 = time.time()
#     best_delta_dict = {}
#     margin = {}

#     surrogate.model.to(device)
#     surrogate.model.eval()
#     # for token in tqdm(node_tokens):
#     alpha_init = 1.0
#     alpha_final = alpha_init / 100
#     gamma_init = 0.05
#     gamma_final = 0.001

#     # true_label = data.y[node_token]
#     # consider the previous prediction as true label
#     true_label = torch.tensor(prediction, device=device)

#     ori_adj = data.adjacency_matrix().to(device)
#     ori_features = data.x.to(device)
#     # labels = data.y.to(device)

#     adj_changes_org = nn.Parameter(torch.FloatTensor(data.num_nodes)).unsqueeze(0).to(device)
#     adj_changes_org[0].data.fill_(0.0001)
#     if args.explore:
#         with torch.no_grad():
#             adj_changes_org[0, init_state[node_token]] = 0.001
#     adj_changes = adj_changes_org * adj_changes_org
#     complementary = None
#     epsilon = torch.ones(batch_size, device=device)

#     worst_norm = torch.max(adj_changes, 1 - adj_changes).norm(p=norm, dim=1)
#     best_norm = worst_norm.clone()

#     best_delta = None
#     adv_found = False

#     logits = surrogate.model(ori_features, utils.normalize(ori_adj.to_dense()))[node_token]
#     posterior = F.softmax(logits, dim=0)
#     outputs = F.log_softmax(logits, dim=0)
#     outputs[true_label] = -1000
#     pred_max_c = outputs.argmax()
#     p_c = posterior[pred_max_c].item()
#     p_y = posterior[true_label].item()

#     # if p_c - p_y > gamma:
#     #     best_norm = 0
#     #     bound.append(0)
#     #     acc_list.append(True)
#     #     ptb_list.append(None)
#     #     best_delta_dict[node_token] = None
#     #     margin[node_token] = p_c - p_y
#     #     print('!!!', posterior, true_label)
#     #     # continue
#     #     # raise ValueError(f'Expect p_c - p_y < gamma, but get p_c({p_c}) and p_y({p_y}).')
#     #     return None

#     if args.explore:
#         index_adv = torch.LongTensor([[node_token, init_state[node_token]], [init_state[node_token], node_token]])
#         values_adv = torch.FloatTensor([-2 * ori_adj[node_token, init_state[node_token]] + 1,
#                                         -2 * ori_adj[init_state[node_token], node_token] + 1])
#         adj_tmp = torch.sparse.FloatTensor(index_adv, values_adv, torch.Size([data.num_nodes, data.num_nodes])).to(device)
#         logits = surrogate.model(ori_features, utils.normalize((ori_adj + adj_tmp).to_dense()))[node_token]
#         posterior = F.softmax(logits, dim=0)
#         output = F.log_softmax(logits, dim=0)
#         output[true_label] = -1000
#         pred_max_c = logits.argmax() 
#         p_y = posterior[true_label].item()
#         p_c = posterior[pred_max_c].item()
#         if p_c - p_y > gamma:
#             best_norm = 1.0
#             bound.append(1.0)
#             acc_list.append(True)
#             ptb_list.append((node_token, init_state[node_token]))
#             best_delta = adj_tmp
#             best_delta_dict[node_token] = best_delta
#             margin[node_token] = p_c - p_y
#             triggers = utils.to_directed(adj_tmp.coalesce().indices().detach().cpu())
#             return list(map(tuple, triggers.t().tolist()))
#             # raise ValueError(f'Expect p_c - p_y < gamma, but get p_c({p_c}) and p_y({p_y}).')

#     t = 0
#     epochs = 800
#     patience = 800
#     L0_list = []

#     while True:
#         if adv_found:
#             patience -= 1
#         if patience == 0:
#             break
    
#         alpha = alpha_init
#         gamma = gamma_init
#         if adv_found and decay:
#             cosine = (1 + math.cos(math.pi * (epochs - patience) / epochs)) / 2
#             alpha = alpha_final + (alpha_init - alpha_final) * cosine
#             gamma = gamma_final + (gamma_init - gamma_final) * cosine
#             alpha = alpha_final + (alpha_init - alpha_final) * cosine
#             gamma = gamma_final + (gamma_init - gamma_final) * cosine

#         delta_norm = adj_changes.data.norm(p=norm, dim=1)
#         L0_list.append(delta_norm)
#         modified_adj = get_modified_adj(complementary, adj_changes, ori_adj.to_dense(), node_token, device)
#         adj_norm = utils.normalize(modified_adj)
#         logits = surrogate.model(ori_features, adj_norm)[node_token].unsqueeze(0)
#         pred_labels = logits.argmax(dim=1)
#         if t == 0:
#             labels_infhot = torch.zeros_like(logits.detach()).scatter(1, true_label.view(1, 1), float('inf'))
#             logit_diff_func = partial(difference_of_logits, labels=true_label, labels_infhot=labels_infhot)
#         loss = -logit_diff_func(logits=logits)
#         delta_grad = grad(loss.sum(), adj_changes_org, only_inputs=True)[0]

#         if t > 0:
#             cols = adj_changes.data.nonzero()[:, 1].tolist()
#             rows = [node_token] * len(cols)
#             ori_values = (-2 * ori_adj[node_token].to_dense()[cols].cpu().numpy() + 1).tolist()
#             index_adv = torch.LongTensor([rows, cols])
#             values_adv = torch.FloatTensor(ori_values)
#             adj_tmp = torch.sparse.FloatTensor(index_adv, values_adv, torch.Size([data.num_nodes, data.num_nodes])).to(device)
#             adj_tmp = adj_tmp + adj_tmp.t()
#             logits_hard = surrogate.model(ori_features, utils.normalize((ori_adj + adj_tmp).to_dense()))[node_token]
#             post_hard = F.softmax(logits_hard, dim=0)
#             output_hard = F.log_softmax(logits_hard, dim=0)
#             # p_y = math.exp(logits_hard[data.labels[target_node]].item())
#             p_y = post_hard[true_label].item()
#             output_hard[true_label] = -1000
#             pred_max_c_hard = output_hard.argmax()
#             p_c = post_hard[pred_max_c_hard].item()
#             is_adv = torch.BoolTensor([p_c - p_y > gamma]).to(device)
#             is_smaller = delta_norm < best_norm
#             is_both = is_adv & is_smaller
#             adv_found = (adv_found + is_adv) > 0
#             best_norm = torch.where(is_both, delta_norm, best_norm)
#             best_delta = adj_tmp.data if is_both[0] else best_delta
#             if is_both[0]:
#                 margin[node_token] = p_c - p_y

#             if delta_norm < 2 and is_adv == True:
#                 break

#             if norm == 0:
#                 epsilon = torch.where(is_adv,
#                                 torch.min(torch.min(epsilon - 1, (epsilon * (1 - gamma)).long().float()), best_norm),
#                                 torch.max(epsilon + 1, (epsilon * (1 + gamma)).long().float()))
#                 epsilon.clamp_min_(0)
#             else:
#                 distance_to_boundary = loss.detach().abs() / delta_grad.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
#                 epsilon = torch.where(is_adv,
#                                 torch.min(epsilon * (1 - gamma), best_norm),
#                                 torch.where(adv_found, epsilon * (1 + gamma), delta_norm + distance_to_boundary))

#             # clip epsilon
#             epsilon = torch.min(epsilon, worst_norm)

#         # normalize gradient
#         grad_l2_norm = delta_grad.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
#         delta_grad.div_(grad_l2_norm.view(batch_size, 1))

#         # gradient ascent step
#         adj_changes_org.data.add_(delta_grad, alpha=alpha)
#         adj_changes = torch.tanh(tanh * adj_changes_org * adj_changes_org)

#         # project
#         adj_changes.data = l0_projection_0(adj_changes, epsilon)
#         t += 1
    
#     # print('adj changes', adj_changes.size(), adj_changes.squeeze())
#     # print('!!!!', adj_changes.nonzero(), adj_changes[0, adj_changes.nonzero()[:, 1]])
#     nodes = adj_changes.squeeze().nonzero().cpu().detach().squeeze().tolist()
#     # print('nodes1', nodes)
#     # nodes = best_delta.to_dense().squeeze().nonzero().cpu().detach().squeeze().tolist()
#     # print('nodes2', nodes)
#     if isinstance(nodes, list):
#         E_t = [(node_token, v) for v in nodes]
#     elif isinstance(nodes, int):
#         E_t = [(node_token, nodes)]
#     else:
#         raise ValueError(f'Invalid nodes: {nodes}')
    
#     # print('best norm:', best_norm)
#     # print('best delta:', best_delta)

#     return E_t
#     # bound.append(best_norm.item())
#     # epoch_list.append(t)
#     # best_delta_dict[node_token] = best_delta
