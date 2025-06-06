import logging
import math
import time
import copy
import random
import numpy as np
import cvxpy as cp
from tqdm  import tqdm
import torch
from torch.distributions.bernoulli import Bernoulli
from statsmodels.stats.proportion import proportion_confint

import utils
from model.gnn import GNN
from robust_gcn_structure.certification import Problem

# logger = logging.getLogger(__name__)

# def find_fragile_nodes(args, surrogate, data, v, label, alpha, d, device):
#     """
#     Find the fragile nodes in the graph.
#     :param model: the model to be used
#     :param data: the data to be used
#     :param alpha: the alpha value
#     :param d: the d value
#     :return: the list of fragile nodes
#     """
#     counts_selection = _sample_noise(args, surrogate, data, v, d, device)
#     d_ca = counts_selection[label].item()
#     p_a_bar = proportion_confint(d_ca, d, alpha=alpha, method='beta')[1]

#     fragile_classes = {}
#     for c in range(data.num_classes):
#         if c == label:
#             continue
#         d_c = counts_selection[c]
#         p_c_bar = proportion_confint(d_c, d, alpha=alpha, method='beta')[0]
#         if p_c_bar >= p_a_bar:
#             fragile_classes[c] = p_c_bar
#     sorted_fragile_classes = {k: v for k, v in sorted(fragile_classes.items(), key=lambda item: item[1], reverse=True)}
#     return len(fragile_classes) > 0, list(sorted_fragile_classes.keys())

relu = lambda x: np.maximum(0, x)

def extract_submatrix(An, ixs, ixs2=None):
    # Slice the input matrix according to the input indices.
    if ixs2 is None:
        ixs2 = ixs
    selected = An[ixs][:, ixs2]
    return selected

def gcn_forward(An, X, weights, i=None, c=None):
    # simple function for a forward pass through GCN.
    if len(weights) == 2:
        W1, W2 = weights
        logits = An @ relu(An @ X @ W1) @ W2
    elif len(weights) == 6:
        W1, b1, W2, b2, W3, b3 = weights
        logits = An @ relu(An @ relu(An @ X @ W1 + b1) @ W2 + b2) @ W3 + b3
    else:
        W1, b1, W2, b2 = weights
        logits = An @ relu(An @ X @ W1 + b1) @ W2 + b2

    # logits = An @ relu(An @ X @ W1 + b1) @ W2 + b2
    logits = logits
    if i is not None:
        logits = logits[i] 
    if c is not None:
        logits = c.T @ logits

        
    return logits

def bounds(problems):
    upper_bounds = np.array([p.upper_bound for p in problems if hasattr(p, "upper_bound")])
    lower_bounds = np.array([p.value for p in problems if hasattr(p, "value")])
    return lower_bounds, upper_bounds


def purge(problems, best_upper, constant=0):
    new_probs = []
    for p in problems:
        if p.open and p.value + constant - best_upper < 1e-8:
            new_probs.append(p)
        if not p.open:
            new_probs.append(p)
    return new_probs

# class CertifiedFragileness(object):
#     """
#     A class to find k-perturbation fragileness nodes
    
#     """

#     def __init__(self, args, model: torch.nn.Module, data, device, 
#                  tolerance=1e-2, max_iter=50, verbose=False) -> None:
#         self.args = args
#         self.model = model
#         self.data = data
#         self.device = device
#         self.tolerance = tolerance
#         self.max_iter = max_iter
#         self.verbose = verbose

#         theta = model.parameters()
#         if len(theta) == 4:
#             self.W1, self.b1, self.W2, self.b2 = theta
#             self.W1 = self.W1.detach().to(self.device)
#             self.b1 = self.b1.detach().to(self.device)
#             self.W2 = self.W2.detach().to(self.device)
#             self.b2 = self.b2.detach().to(self.device)
#             self.bias = True
#         elif len(theta) == 2:
#             self.W1, self.W2 = theta
#             self.W1 = self.W1.detach().to(self.device)
#             self.W2 = self.W2.detach().to(self.device)
#             self.bias = False
#         self.theta = (self.W1, self.b1, self.W2, self.b2) if self.bias else (self.W1, self.W2)  

#         self.X = self.data.x.to(self.device)
#         self.adj = self.data.adjacency_matrix().to_dense().to(self.device)
#         self.adj_tilde = self.adj + torch.eye(self.data.num_nodes, device=self.device)
#         self.adj_norm = utils.normalize(self.adj_tilde)
#         self.degree = self.adj_tilde.sum(dim=1).flatten()
#         self.deg_matrix = torch.diag(self.degree)
#         self.deg_matrix_norm = torch.pow(self.deg_matrix, -0.5)
#         self.deg_matrix_norm[torch.isinf(self.deg_matrix_norm)] = 0.

#     def certify(self, target_node, correct_label, target_label, candidates=None, solver='ECOS'):
#         """
#         v: the target node to certify.
#         correct_label: y*, the label of the target node predicted by target GNN.
#         target_label: y, the label of the target node to certify.
#         """
#         if self.verbose:
#             print('target node:', target_node, ', y*:', correct_label, ', y:', target_label, ', candidate:', candidates) 

#         assert correct_label != target_label, 'The target label should be different from the correct label.'
#         if candidates is not None:
#             target_candidates = [target_node] + candidates
            
#             neighbors = self.data.neighbors(target_candidates, l=self.args.candidate_hop)
#             # merge the two lists
#             neighbors = np.concatenate((target_candidates, neighbors)).tolist()

#             node2idx = {u: i for i, u in enumerate(neighbors)}
#             idx2node = {i: u for i, u in enumerate(neighbors)}

#             self._degree = self.degree[neighbors]
#             self._adj = self.adj[neighbors][:, neighbors]
#             self._adj_tilde = self.adj_tilde[neighbors][:, neighbors]
#             self._adj_norm = self.adj_norm[neighbors][:, neighbors]
#             self._X = self.X[neighbors]

#             target_idx = node2idx[target_node]
#             candidates_idx = [node2idx[u] for u in candidates]

#             if self.verbose:
#                 print('degree:', self._degree)
#                 print('adj:', self._adj)
#                 print('adj_norm:', self._adj_norm)
#         else:
#             target_idx = target_node
#             self._degree = self.degree.clone()
#             self._adj = self.adj.clone()
#             self._adj_tilde = self.adj_tilde.clone()
#             self._adj_norm = self.adj_norm.clone()
#             self._X = self.X.clone()
#             neighbors = []

#         t0 = time.time()
#         XW = torch.mm(self._X, self.W1)
#         # create two variables for the perturbation (positive changes and negative changes)
#         epsilon_plus = cp.Variable(shape=self._adj.shape, name='epsilon_plus', symmetric=True)
#         epsilon_minus = cp.Variable(shape=self._adj.shape, name='epsilon_minus', symmetric=True)

#         # convex variable for the adjacency matrix, and adj_norm with target node
#         adj_norm_cvx = self._adj_norm.cpu().numpy() - epsilon_minus + epsilon_plus
#         adj_norm_v = self._adj_norm[target_idx].cpu().numpy() - epsilon_minus[target_idx] + epsilon_plus[target_idx]

#         if self.bias:
#             H2_hat_cvx = adj_norm_cvx @ XW.cpu().numpy() + np.tile(self.b1.cpu().numpy(), [adj_norm_cvx.shape[0], 1])
#             H2_hat = (torch.mm(self._adj_norm, XW) + self.b1).cpu().numpy()
#         else:
#             H2_hat_cvx = adj_norm_cvx @ XW.cpu().numpy()
#             H2_hat = (torch.mm(self._adj_norm, XW)).cpu().numpy()
        
#         H2 = relu(H2_hat)
#         H2_cvx = cp.Variable(shape=(H2_hat.shape[0], H2_hat.shape[1]), name="H2", nonneg=True)
#         H2_cvx.value = H2 

#         c = self._compute_c(correct_label, target_label)
#         try:
#             optimization_problem, element_lower, element_upper, row_lower, row_upper, row_l1_distance_upper, global_lower, global_upper, pos_lower, pos_upper = self._build_optimization_problem(c, target_idx, candidates_idx, adj_norm_v, adj_norm_cvx, epsilon_minus, epsilon_plus, XW, H2_hat_cvx, H2_cvx)
#         except ValueError as e:
#             # print('found the case,', e)
#             # print('target:', target_node, ', candidates:', candidates)
#             # print('target_candidates', target_candidates)
#             # print('degree:', self._degree[:4])
#             # print('degree2:', self.degree[target_candidates])
#             # print('adj_norm:', self._adj_norm[[0] + candidates_idx][:, [0] + candidates_idx].cpu())
#             # print('adj_norm2:', self.adj_norm[target_candidates][:, target_candidates].cpu())
#             raise e
#         except AssertionError as e:
#             # print('target:', target_node, ', candidates:', candidates)
#             # print('target_candidates', target_candidates)
#             # print('degree:', self._degree[:4])
#             raise e
#         # print('build optimization problem time:', time.time() - t0)
#         prob_built_time = time.time() - t0

#         try:
#             prob_state = optimization_problem.solve(solver=solver)
#             if prob_state == 'infeasible':
#                 # print(f'  ==> solving {target_node} is infeasible')
#                 # print('degree:', self._degree)
#                 # print('adj_norm:', self._adj_norm)
#                 # print('element_upper:', element_upper)
#                 # print('element_lower:', element_lower)
#                 # print('row_lower:', row_lower)
#                 # print('row_upper:', row_upper)
#                 # print('row_l1_distance_upper:', row_l1_distance_upper)
#                 # print('global_bounds:', global_lower, global_upper)
#                 # print('pos_bounds:', pos_lower, pos_upper)
#                 return {'fragile': False, 'error': True}

#             problems = [optimization_problem]
#             best_uppers = [optimization_problem.upper_bound + (c.T @ self.b2.cpu().numpy())[0]]
#             best_lowers = [optimization_problem.value + (c.T @ self.b2.cpu().numpy())[0]]
#             solve_times = [optimization_problem.problem.solver_stats.solve_time]

#             worst_lower = best_lowers[0]
#             logit_diff_before = gcn_forward(self.adj_norm, self.X, self.theta, target_node, c).squeeze().item()

#             # robust = False
#             fragile = False
#             to_branch = optimization_problem
#             for step in range(int(self.max_iter)):
#                 lower_bounds, upper_bounds = bounds(problems)
#                 if self.bias:
#                     best_upper_bound = np.min(upper_bounds) + (c.T @ self.b2.cpu().numpy())[0]
#                 else:
#                     best_upper_bound = np.min(upper_bounds)

#                 if step > 0:
#                     best_uppers.append(best_upper_bound)

#                 if best_upper_bound < 0:
#                     fragile = True
#                     break

#                 if self.bias:
#                     problems = purge(problems, best_upper_bound, constant=(c.T @ self.b2.cpu().numpy())[0])
#                 else:
#                     problems = purge(problems, best_upper_bound)

#                 open_problems = [p for p in problems if p.open]
#                 if len(open_problems) == 0:
#                     fragile = worst_lower < 0
#                     # print('break from no open problems!')
#                     break

#                 if self.bias:
#                     worst_lower_n = np.min([p.value for p in open_problems]) + (c.T @ self.b2.cpu().numpy())[0]
#                 else:
#                     worst_lower_n = np.min([p.value for p in open_problems])

#                 # assert worst_lower_n - worst_lower >= -1e-10
#                 worst_lower = worst_lower_n
#                 if step > 0:
#                     best_lowers.append(worst_lower)

#                 if worst_lower > 0:
#                     fragile = False
#                     # print('break from worst_lower > 0!')
#                     break

#                 if best_upper_bound - worst_lower < self.tolerance:
#                     fragile = False
#                     # print('break from best_upper_bound - worst_lower < self.tolerance!')
#                     break

#                 to_branch = open_problems[np.argmin([p.value for p in open_problems])]
#                 to_branch.branch()
#                 for child in to_branch.children:
#                     child.solve(solver=solver)
#                     solve_times.append(child.problem.solver_stats.solve_time)
#                     problems.append(child)

#             best_prob = to_branch
#             best_adj_pert = self._adj_norm.cpu().numpy() - best_prob.vars_opt['epsilon_minus'] + best_prob.vars_opt['epsilon_plus']

#             # best_adj_pert - self._adj_norm.cpu().numpy()
#             _mask = torch.where(self._adj == 1, torch.zeros_like(self._adj), torch.ones_like(self._adj)).cpu().numpy()
#             potential_perturbations = np.multiply(_mask, np.triu(np.abs(self._adj_norm.cpu().numpy() - best_adj_pert), 1))
#             # Pick top 5 as the perturbations
#             sorted_idx = np.argsort(potential_perturbations, axis=None)[::-1][:self.args.candidate_size]
#             pert_indices = np.unravel_index(sorted_idx, potential_perturbations.shape)
#             # pert_indices = np.where(potential_perturbations > 0.1)
#             adj_diff = potential_perturbations[pert_indices]
#             perturbations = np.concatenate(pert_indices).reshape(2, -1).T

#             if candidates is not None: # map back to the original node index
#                 perturbations = np.array([(idx2node[i], idx2node[j]) for i, j in perturbations])
#             else:
#                 raise NotImplementedError('Not implemented for the case of one-hop nodes.')
            
#             # print(f'target node: {target_node}, target label: {target_label}, twohop nodes: {twohops}, perturbations: {perturbations}')

#             if self.verbose:
#                 print('fragile:', fragile)
#                 print('best_upper_bound:', best_upper_bound)
#                 # print('adj_norm:', best_adj_pert)
#                 print('perturbations:', perturbations)
#                 print('adj_diff:', adj_diff)

#             return {
#                 'fragile': True if fragile else False,
#                 'fragile_score': best_upper_bound.item(),
#                 'best_uppers': best_uppers,
#                 'best_lowers': best_lowers, 
#                 'adj_pert': best_adj_pert,
#                 # 'best_perturbation': best_prob.vars_opt['epsilon_minus'],
#                 'perturbations': perturbations.tolist(),
#                 'adj_diff': adj_diff.tolist(),
#                 'logit_diff_before': logit_diff_before,
#                 'solve_times': solve_times,
#                 'neighbors': neighbors,
#                 'error': False,
#                 'buit_time': prob_built_time,
#                 'step': step,
#             }

#         except cp.error.SolverError as err:
#             print('SolverError:', err)
#             return {'fragile': False, 'error': True, 'built_time': prob_built_time}


#     def _compute_c(self, correct_label, target_label):
#         c = np.zeros(self.data.num_classes)
#         c[correct_label] = 1
#         c[target_label] = -1
#         c = c[:, None]
#         return c
    
#     def _build_optimization_problem(self, c, target_idx, candidates_idx, adj_norm_v, adj_cvx, epsilon_minus, epsilon_plus, XW, H2_hat_cvx, H2_cvx):
#         domain_constraints, bounds = self._build_domin_constraints(target_idx, candidates_idx, adj_norm_v, adj_cvx, epsilon_minus, epsilon_plus)
#         row_lower, row_upper, row_l1_distance_upper, global_lower, global_upper, pos_lower, pos_upper = bounds

#         # element_lower, element_upper = self._element_bounds()
#         element_lower, element_upper = self._element_bounds_new(target_idx, candidates_idx)

#         if self.verbose:
#             print('element_lower:', element_lower)
#             print('element_upper:', element_upper)

#         # print('check 1', torch.all(self._adj_norm <= element_upper), element_upper[self._adj_norm > element_upper], self._adj_norm[self._adj_norm > element_upper])
        
#         epsilon_plus_upper_bound = element_upper.cpu().numpy() - self._adj_norm.cpu().numpy()
#         epsilon_plus_upper_bound = np.where(epsilon_plus_upper_bound <= 0, 0, epsilon_plus_upper_bound)

#         assert np.all(epsilon_plus_upper_bound >= 0), f'epsilon_plus_upper_bound < 0, {epsilon_plus_upper_bound[epsilon_plus_upper_bound < 0]}'
#         # print('epsilon_plus_upper_bound', epsilon_plus_upper_bound)

#         non_zero_indices = self._adj_norm.cpu().numpy().nonzero()
#         element_constraints = [
#             adj_cvx >= element_lower.cpu().numpy(),
#             adj_cvx <= element_upper.cpu().numpy(),
#             epsilon_minus <= self._adj_norm.cpu().numpy(),
#             epsilon_plus <= epsilon_plus_upper_bound,
#             # epsilon_plus[non_zero_indices] / epsilon_plus_upper_bound[non_zero_indices] \
#             # + epsilon_minus[non_zero_indices] / self._adj_norm.cpu().numpy()[non_zero_indices] <= 1,
#         ]
#         domain_constraints.extend(element_constraints)

#         triangle_constraints, R, S = self._build_triangle_constraints(XW, element_lower, element_upper, H2_hat_cvx, H2_cvx)

#         L_x = element_lower[target_idx].cpu().numpy()
#         U_x = element_upper[target_idx].cpu().numpy()
#         L_y = np.maximum(R, 0)
#         U_y = np.maximum(S, 0)
#         WC = self.W2.cpu().numpy() @ c

#         x_var = adj_norm_v

#         z_lower = (L_y @ relu(WC) - U_y @ relu(-WC))[:, 0]
#         z_upper = (U_y @ relu(WC) - L_y @ relu(-WC))[:, 0]

#         assert np.all(z_lower <= z_upper), f'z_lower < z_upper, {z_lower[z_lower > z_upper]} < {z_upper[z_lower > z_upper]}'
#         z = cp.Variable(z_lower.shape, name='z')
#         z_constraints = [
#             z >= z_lower,
#             z <= z_upper,
#             z == (H2_cvx @ WC)[:, 0]
#         ]
#         constraints = {
#             'domain_constraints': domain_constraints,
#             'triangle_constraints': triangle_constraints,
#             'z_constraints': z_constraints
#         }

#         extra_var_eps_minus = cp.Variable(epsilon_minus[target_idx].shape,
#                                            'extra_var_eps_minus')
#         extra_var_eps_plus = cp.Variable(epsilon_plus[target_idx].shape,
#                                           'extra_var_eps_plus')
#         constraints['extra_constraints'] = [
#             extra_var_eps_minus == epsilon_minus[target_idx],
#             extra_var_eps_plus == epsilon_plus[target_idx],
#         ]

#         prb = Problem(L_x, U_x, z_lower, z_upper, x_var, z, constraints,
#                       x_orig=self._adj_norm[target_idx],
#                       eps_minus_x=extra_var_eps_minus,
#                       eps_plus_x=extra_var_eps_plus)

#         return prb, element_lower, element_upper, row_lower, row_upper, row_l1_distance_upper, global_lower, global_upper, pos_lower, pos_upper


#     def _build_domin_constraints(self, target_idx, candidates_idx, adj_v_cvx, adj_cvx, epsilon_minus, epsilon_plus):
#         row_lower, row_upper, row_l1_distance_upper = self._row_bounds_new(target_idx, candidates_idx)
#         # row_lower, row_upper, row_l1_distance_upper = self._row_bounds()
#         global_lower, global_upper = self._global_bound(target_idx, candidates_idx)
#         # assert global_lower <= global_upper, f'global_lower > global_upper, {global_lower} > {global_upper}'
#         pos_lower, pos_upper = self._pos_bounds(target_idx, candidates_idx)
#         assert pos_lower <= pos_upper, f'pos_lower > pos_upper, {pos_lower} > {pos_upper}'
#         if self.verbose:
#             print('row_lower:', row_lower)
#             print('row_upper:', row_upper)
        
#         # mask_plus = np.ones_like(self._adj.cpu().numpy())
#         # _candidates_idx = copy.deepcopy(candidates_idx)
#         # _candidates_idx.remove(target_idx)
#         # indices = ([target_idx] * len(candidates_idx) + candidates_idx, candidates_idx + [target_idx] * len(candidates_idx))
#         # mask_plus[indices] = 0

#         # mask_minus = np.ones_like(self._adj.cpu().numpy())
#         # mask_minus[np.where(self._adj_tilde.cpu().numpy() == 1)] = 0

#         domain_constraints = [
#             cp.diag(epsilon_plus) == 0,
#             epsilon_minus >= 0,
#             epsilon_plus >= 0,

#             # only entries without (=0) can be perturbed
#             # cp.multiply(self._adj_tilde.cpu().numpy(), epsilon_minus) == 0,
#             # cp.multiply(self._adj_tilde.cpu().numpy(), epsilon_plus) == 0,

#             # only allow perturbation between the target node and candidate nodes
#             # cp.multiply(mask_plus, epsilon_plus) == 0,
#             # only entries == 1 can be reduced
#             # cp.multiply(mask_minus, epsilon_minus) == 0,

#             cp.sum(epsilon_plus) >= pos_lower.cpu().numpy() * 2,
#             cp.sum(epsilon_plus) <= pos_upper.cpu().numpy() * 2,
#             # cp.sum(epsilon_plus) <= 1,

#             # tie together the row of node v with the matrix variable
#             adj_v_cvx == adj_cvx[target_idx],

#             # row-wise bounds
#             cp.sum(adj_cvx, axis=1) >= row_lower.cpu().numpy(),
#             cp.sum(adj_cvx, axis=1) <= row_upper.cpu().numpy(),

#             # row-wise bounds II
#             cp.sum(cp.abs(adj_cvx - self._adj_norm.cpu().numpy()), axis=1) <= row_l1_distance_upper.cpu().numpy(),
            
#             # global upper bound
#             # cp.sum(cp.abs(adj_cvx - self._adj_norm.cpu().numpy())) >= global_lower.cpu().numpy(),
#             cp.sum(cp.abs(adj_cvx - self._adj_norm.cpu().numpy())) <= global_upper.cpu().numpy() * 2,
#             # cp.sum(cp.abs(adj_cvx - self._adj_norm.cpu().numpy())) <= 2,
#         ]
#         return domain_constraints, (row_lower, row_upper, row_l1_distance_upper, None, global_upper, pos_lower, pos_upper)
    
#     def _get_area(self, i, j, target_idx, candidates_idx):
#         if i == target_idx and (j == target_idx or j in candidates_idx):
#             return 1
#         elif i == target_idx and j not in candidates_idx:
#             return 2
#         elif i in candidates_idx:
#             return 3
#         elif i not in candidates_idx and j not in candidates_idx:
#             return 4

    
#     def _element_bounds_new(self, target_idx, candidates_idx):
#         """ we assume that there is exactly one perturbation between the target node and candidate nodes
#         """
#         # print('target_idx:', target_idx)
#         # print('candidates_idx:', candidates_idx)

#         upper = torch.zeros_like(self._adj)
#         lower = torch.zeros_like(self._adj)
#         for i in range(self._adj.size(0)):
#             for j in range(i, self._adj.size(0)):
#                 area = self._get_area(i, j, target_idx, candidates_idx)
#                 # print(i, j, ', area:', area)
#                 if area == 1:
#                     if i == j: # diagonal
#                         lower[i, j] = 1 / (self._degree[i] + 1)
#                         upper[i, j] = 1 / (self._degree[i])
#                     else:
#                         lower[i, j] = 0
#                         upper[i, j] = 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[j])
#                 elif area == 2:
#                     lower[i, j] = torch.minimum(self._adj_norm[i, j], 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[j]))
#                     upper[i, j] = torch.maximum(self._adj_norm[i, j], 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[j]))
#                     # upper[i, j] = self._adj_norm[i, j]
#                 elif area == 3:
#                     if i == j:
#                         lower[i, j] = 1 / (self._degree[i] + 1)
#                         upper[i, j] = self._adj_norm[i, j]
#                         # print(f'check {i}-{j}', upper[i, j], self._adj_norm[i, j])
#                         # if upper[i, j] != self._adj_norm[i, j]:
#                         #     print(f'check {i}-{j}', upper[i, j], self._adj_norm[i, j])
#                         #     exit(0)
#                     else:
#                         lower[i, j] = torch.minimum(self._adj_norm[i, j], 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[j] + 1))
#                         upper[i, j] = torch.maximum(self._adj_norm[i, j], 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[j] + 1))
#                 elif area == 4:
#                     lower[i, j] = self._adj_norm[i, j]
#                     upper[i, j] = self._adj_norm[i, j]

#                 if upper[i, j].cpu().item() - self._adj_norm[i, j].cpu().item() < -1e-5:
#                     raise ValueError(f'upper bound is smaller than the original value, at {i}-{j}. {upper[i, j]} < {self._adj_norm[i, j]}')

#         lower = lower + lower.t() - torch.diag(torch.diag(lower))
#         upper = upper + upper.t() - torch.diag(torch.diag(upper))

#         # lower = torch.where(lower == 0, torch.zeros_like(lower) + 1e-5, lower)
#         # upper = torch.where(upper == 0, torch.zeros_like(upper) + 1e-5, upper)

#         assert torch.all(lower <= upper), f'Invalid element bounds, lower is greater than upper. {lower[lower>upper]} > {upper[lower>upper]}.' 
#         return lower, upper

 
#     def _element_bounds(self):
#         # Case 1: 1 / sqrt(d_u + 1) + 1 / sqrt(d_v + 1)
#         delta_deg_mx_i = (self._degree + 1).unsqueeze(1).repeat(1, self._adj.size(0))
#         delta_deg_mx_j = (self._degree + 1).repeat(self._adj.size(0), 1)
#         delta_adj_norm = 1 / torch.sqrt(delta_deg_mx_i) / torch.sqrt(delta_deg_mx_j)

#         upper = torch.maximum(self._adj_norm, delta_adj_norm)
#         lower = torch.where(self._adj == 0, self._adj, delta_adj_norm)

#         assert torch.all(lower <= upper), f'Invalid element bounds, lower is greater than upper. {lower[lower>upper]} > {upper[lower>upper]}.'
#         return lower, upper
    
#     def _row_bounds_new(self, target_idx, candidates_idx):
#         sorted_indices = torch.argsort(self._degree[candidates_idx])
#         smallest_deg_candidate = candidates_idx[sorted_indices[0]]
#         largest_deg_candidate = candidates_idx[sorted_indices[-1]]

#         deg_v = self._degree[target_idx]

#         lower = torch.zeros(self._adj.size(0))
#         upper = torch.zeros(self._adj.size(0))
#         l1_distance_upper = torch.zeros(self._adj.size(0))
#         for i in range(self._adj.size(0)):
#             _neighbors = self._adj[i].nonzero().squeeze(1).tolist()
#             case1_low = torch.sum(1/torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[_neighbors])) \
#                     + 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[largest_deg_candidate] + 1) \
#                     + 1 / (self._degree[i] + 1)  
#             case1_high = torch.sum(1/torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[_neighbors])) \
#                     + 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[smallest_deg_candidate] + 1) \
#                     + 1 / (self._degree[i] + 1)  
#             if i == target_idx:
#                 lower[i] = torch.sum(1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[_neighbors])) \
#                         + 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[largest_deg_candidate] + 1) \
#                         + 1 / (self._degree[i] + 1)
#                 upper[i] = torch.sum(1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[_neighbors])) \
#                         + 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[smallest_deg_candidate] + 1) \
#                         + 1 / (self._degree[i] + 1)
#                 l1_distance_upper[i] = torch.sum((1 / torch.sqrt(self._degree[i]) - 1 / torch.sqrt(self._degree[i] + 1)) / torch.sqrt(self._degree[_neighbors])) \
#                                 + 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[smallest_deg_candidate] + 1)
#             elif i in candidates_idx:
#                 # low[i] = torch.sum(1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[_neighbors])) \
#                 #             + 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(largest_deg_candidate + 1) + 1 / (self._degree[i] + 1)
#                 lower[i] = torch.minimum(self._adj_norm[i].sum(), case1_low)
#                 upper[i] = torch.maximum(self._adj_norm[i].sum(), case1_high)
                
#                 # delta_row_sum = torch.sum(1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[_neighbors])) \
#                 #             + 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(deg_v + 1) + 1 / (self._degree[i] + 1)
#                 # lower[i] = torch.minimum(self._adj_norm[i].sum(), delta_row_sum)
#                 # upper[i] = torch.maximum(self._adj_norm[i].sum(), delta_row_sum)

#                 l1_distance_upper[i] = torch.sum((1 / torch.sqrt(self._degree[i]) - 1 / torch.sqrt(self._degree[i] + 1)) / torch.sqrt(self._degree[_neighbors])) \
#                                 + 1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(deg_v + 1)
#             else:
#                 lower[i] = torch.minimum(self._adj_norm[i].sum(), case1_low)
#                 upper[i] = torch.maximum(self._adj_norm[i].sum(), case1_high)
#                 # _degree = self._degree.clone()
#                 # _degree[[target_idx] + candidates_idx] += 1 
#                 # lower[i] = torch.sum(1 / torch.sqrt(_degree[i]) / torch.sqrt(_degree[_neighbors])) + 1 / self._degree[i]
#                 # upper[i] = self._adj_norm[i].sum()
#                 l1_distance_upper[i] = 1 /torch.sqrt(self._degree[i]) / torch.sqrt(deg_v) -  1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(deg_v + 1) \
#                                 + 1 / torch.sqrt(self._degree[i]) / torch.sqrt(self._degree[smallest_deg_candidate]) - 1 / torch.sqrt(self._degree[i]) / torch.sqrt(self._degree[smallest_deg_candidate] + 1)
        
#         assert torch.all(lower - upper < 1e5), f'Invalid row bounds, lower is greater than upper. {lower[lower>upper]} > {upper[lower>upper]}, at {torch.where(lower > upper)}.'
#         lower = torch.where(lower > upper, upper, lower)

#         return lower, upper, l1_distance_upper

    
#     def _row_bounds(self):
#         """ Eqn (6) in the paper
#         """
#         u_deg, v_deg = torch.sort(self._degree).values[:2] # the two smallest degrees
#         upper = torch.sum(self._adj_norm, dim=1) + 1 / torch.sqrt(self._degree * (u_deg +1)) + 1 / torch.sqrt(self._degree * (v_deg + 1))
#         # upper = upper.cpu().numpy()

#         # Case 1 & 2
#         lower = torch.zeros(self._adj.size(0))
#         row_l1_distance_upper = torch.zeros(self._adj.size(0))
#         for i in range(self._adj.size(0)):
#             neighbors = self._adj[i].nonzero().squeeze(1).tolist()

#             _degree = self._degree.clone()
#             _degree[neighbors] = 0 # filter out the neighbors
#             u_deg = torch.sort(_degree).values[-1] # the largest degree
            
#             lower[i] = torch.sum(1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[neighbors])) + (1 / torch.sqrt((self._degree[i] + 1) * (u_deg + 1)))
#             row_l1_distance_upper[i] = 1 / torch.sqrt(self._degree[i]) / torch.sqrt(u_deg + 1) + 1 / torch.sqrt(self._degree[i]) / torch.sqrt(v_deg + 1)
        
#         return lower, upper, row_l1_distance_upper
 
#     def _global_bound(self, target_idx, candidates_idx):
#         """ Eqn (9) in the paper
#         """
#         d_a = self._degree[target_idx]
#         idx_max = torch.argmax(self._degree[candidates_idx])
#         idx_min = torch.argmin(self._degree[candidates_idx])
#         d_max = self._degree[candidates_idx][idx_max]
#         d_min = self._degree[candidates_idx][idx_min]
#         # print('d:', d_a, d_max, d_min)
#         # print('idx:', idx_max, idx_min)

#         _delta_adj_norm_max = self._adj_norm[idx_max].clone()
#         _delta_adj_norm_max[idx_max] = 0
#         lower = 4 * (1 / d_max - 1 / (d_max + 1)) + 2 * torch.sum(_delta_adj_norm_max * (1 / torch.sqrt(d_max) - 1 / torch.sqrt(d_max + 1)))
#         _delta_adj_norm_min = self._adj_norm[idx_min].clone()
#         _delta_adj_norm_min[idx_min] = 0
#         upper = 4 * (1 / d_min - 1 / (d_min + 1)) + 2 * torch.sum(_delta_adj_norm_min * (1 / torch.sqrt(d_min) - 1 / torch.sqrt(d_min + 1)))

#         # if d_max == d_min:
#         #     idx_min = len(candidates_idx) - 1

#         # adj_norm_max = copy.deepcopy(self._adj_norm)
#         # adj_norm_min = copy.deepcopy(self._adj_norm)
#         # for i,j in torch.nonzero(torch.triu(self._adj_norm, diagonal=0)):
#         #     if i == target_idx and j == target_idx:
#         #         adj_norm_max[i, j] = 1 / (d_a + 1)
#         #         adj_norm_min[i, j] = 1 / (d_a + 1)
#         #     elif i == candidates_idx[idx_max] and j == candidates_idx[idx_max]:
#         #         adj_norm_max[i, j] = 1 / (d_max + 1)
#         #     elif i == candidates_idx[idx_min] and j == candidates_idx[idx_min]:
#         #         adj_norm_min[i, j] = 1 / (d_min + 1)
#         #     elif i == target_idx and j == candidates_idx[idx_max]:
#         #         adj_norm_max[i, j] = 1 / torch.sqrt(d_a + 1) / torch.sqrt(d_max + 1)
#         #     elif i == target_idx and j == candidates_idx[idx_min]:
#         #         adj_norm_min[i, j] = 1 / torch.sqrt(d_a + 1) / torch.sqrt(d_min + 1)
#         #     elif i == target_idx:
#         #         adj_norm_max[i, j] = 1 / torch.sqrt(d_a + 1) / torch.sqrt(self._degree[j])
#         #         adj_norm_min[i, j] = 1 / torch.sqrt(d_a + 1) / torch.sqrt(self._degree[j])
#         #     elif i == candidates_idx[idx_max]:
#         #         adj_norm_max[i, j] = 1 / torch.sqrt(d_max + 1) / torch.sqrt(self._degree[j])
#         #     elif i == candidates_idx[idx_min]:
#         #         adj_norm_min[i, j] = 1 / torch.sqrt(d_min + 1) / torch.sqrt(self._degree[j])

#         # lower = torch.abs(torch.triu(adj_norm_max, 0) - torch.triu(self._adj_norm, diagonal=0)).sum() * 2
#         # upper = torch.abs(torch.triu(adj_norm_min, 0) - torch.triu(self._adj_norm, diagonal=0)).sum() * 2
#         # print('check global bound', lower, upper)
#         # print('adj_norm_max', adj_norm_max)
#         # print('adj_norm_min', adj_norm_min)
#         # print('adj_norm', self._adj_norm)
#         # if d_max == d_min:
#         #     lower = torch.minimum(lower, upper)
#         #     upper = torch.maximum(lower, upper)
#         # exit(0)

#         return lower, upper
    
#     def _pos_bounds(self, target_idx, candidates_idx):
#         d_a = self._degree[target_idx]
#         d_max = torch.max(self._degree[candidates_idx])
#         d_min = torch.min(self._degree[candidates_idx])
#         lower = 1 / torch.sqrt(d_a + 1) / torch.sqrt(d_max + 1)
#         upper = 1 / torch.sqrt(d_a + 1) / torch.sqrt(d_min + 1)
#         return lower, upper
        

#     def _relu_triangle_relaxation(self, XW, element_lower, element_upper):
#         H2_hat = self._adj_norm @ self._X @ self.W1
#         if self.bias:
#             H2_hat = H2_hat + self.b1

#         XW_plus = torch.maximum(XW, torch.zeros_like(XW))
#         XW_minus = -torch.minimum(XW, torch.zeros_like(XW))

#         S = element_upper @ XW_plus - element_lower @ XW_minus
#         R = element_lower @ XW_plus - element_upper @ XW_minus
#         if self.bias:
#             S = S + self.b1
#             R = R + self.b1

#         assert torch.all(R <= S), f'R <= S, {R[R > S]} <= {S[R > S]}'

#         I = (S > 0) & (R < 0)
#         I_plus = (S >= 0) & (R >= 0)
#         I_minus = (S <= 0) & (R <= 0)
#         # if not torch.all(H2_hat - S <= 1e-5):
#         #     torch.set_printoptions(precision=4, sci_mode=False)
#         #     print('H2_hat - S <= 1e-5')
#         #     print('H2_hat:', H2_hat[H2_hat - S > 1e-5])
#         #     print('S:', S[H2_hat - S > 1e-5])
#         #     print('at:', torch.where(H2_hat - S > 1e-5))
#         #     print('element_upper:', element_upper)
#         #     print('element_lower:', element_lower)
#         #     print('adj_norm:', self._adj_norm)
#         #     print('degree:', self._degree)
#         #     exit(0)

#         assert torch.all(I + I_plus + I_minus)
#         assert (H2_hat - S <= 1e-5).all(), f'at {torch.where(H2_hat - S > 1e-5)}, upper_bound {element_upper[torch.where(H2_hat - S > 1e-5)]}, adj_norm: {self._adj_norm[torch.where(H2_hat - S > 1e-5)]}.'
#         assert (H2_hat - R >= -1e-5).all(), f'H2_hat: {H2_hat[H2_hat - R < -1e-5]} < R: {R[H2_hat - R < -1e-5]}, at {torch.where(H2_hat - R < -1e-5)}'
#         S_I = torch.multiply(S, I)
#         R_I = torch.multiply(R, I)

#         assert torch.all(R_I <= S_I), f'R_I <= S_I, {R_I[R_I > S_I]} <= {S_I[R_I > S_I]}'
#         return I, I_minus, I_plus, R, R_I, S, S_I

    
#     def _build_triangle_constraints(self, XW, element_lower, element_upper, H2_hat_cvx, H2_cvx):
#         I, I_minus, I_plus, R, R_I, S, S_I = self._relu_triangle_relaxation(XW, element_lower, element_upper)
        
#         H2_cvx_I = cp.multiply(I.cpu().numpy(), H2_cvx)
#         H2_cvx_I_plus = cp.multiply(I_plus.cpu().numpy(), H2_cvx)
#         H2_cvx_I_minus = cp.multiply(I_minus.cpu().numpy(), H2_cvx)
#         H2_hat_cvx_I = cp.multiply(I.cpu().numpy(), H2_hat_cvx)
#         H2_hat_cvx_I_plus = cp.multiply(I_plus.cpu().numpy(), H2_hat_cvx)
#         triangle_constraints = [
#             H2_cvx >= H2_hat_cvx,
#             H2_cvx_I_minus == 0,
#             H2_cvx_I_plus == H2_hat_cvx_I_plus,
#             cp.multiply(S_I.cpu().numpy() - R_I.cpu().numpy(), H2_cvx_I) <= cp.multiply(S_I.cpu().numpy(), H2_hat_cvx_I - R_I.cpu().numpy())
#         ]
#         return triangle_constraints, R.cpu().numpy(), S.cpu().numpy()
                           
#     def _deg_norm(self, i_deg, j_deg):
#         return 1 / torch.sqrt(i_deg) / torch.sqrt(j_deg)


class CertifiedFragilenessLazy(object):
    """
    A class to find k-perturbation fragileness nodes
    
    """
    def __init__(self, args, weights, data, num_layer=2,
                 tolerance=1e-2, max_iter=50, verbose=False, deletion=None) -> None:
        self.args = args
        self.data = data
        self.num_layer = num_layer
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.verbose = verbose

        # print('weights:', [p.shape for p in weights])
        # print('num_layer:', num_layer, ', weights:', len(weights))

        if len(weights) == 4:
            self.W1, self.b1, self.W2, self.b2 = weights
            self.W1 = self.W1
            self.b1 = self.b1
            self.W2 = self.W2
            self.b2 = self.b2
            self.theta = (self.W1, self.b1, self.W2, self.b2)
            self.bias = True
        elif len(weights) == 2 and self.num_layer == 2:
            self.W1, self.W2 = weights
            self.W1 = self.W1
            self.W2 = self.W2
            self.theta = (self.W1, self.W2)
            self.bias = False
        elif len(weights) == 2 and self.num_layer == 1:
            self.W1, self.b1 = weights
            self.bias = True
            self.theta = (self.W1, self.b1)
        elif len(weights) == 6 and self.num_layer == 3:
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = weights
            self.bias = True
            self.theta = (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)

        self.deletion = False
        if deletion is not None:
            self.del1, self.del2 = deletion
            self.deletion = True

        self.X = self.data.x.numpy()
        self.adj = self.data.adjacency_matrix().to_dense().numpy()
        self.adj_tilde = self.adj + np.eye(self.data.num_nodes)
        self.adj_norm = utils.preprocess_adj(self.adj)
        self.degree = self.adj_tilde.sum(axis=1).flatten()
        self.deg_matrix = np.diag(self.degree)
        with np.errstate(divide='ignore'):
            self.deg_matrix_norm = np.power(self.deg_matrix, -0.5)
        self.deg_matrix_norm[np.isinf(self.deg_matrix_norm)] = 0.

    def certify(self, target_node, correct_label, target_label, candidates=None, solver='ECOS', mask=None):
        """
        v: the target node to certify.
        correct_label: y*, the label of the target node predicted by target GNN.
        target_label: y, the label of the target node to certify.
        """
        if self.verbose:
            print('target node:', target_node, ', y*:', correct_label, ', y:', target_label, ', candidate:', candidates) 
        assert correct_label != target_label, 'The target label should be different from the correct label.'
        self.target_label = target_label
        if candidates is not None:
            target_candidates = [target_node] + candidates
            
            neighbors = self.data.neighbors(target_candidates, l=self.args.candidate_hop)
            # merge the two lists
            neighbors = np.concatenate((target_candidates, neighbors)).tolist()

            # print('the number of target + candidates + neighbors:', len(neighbors))

            node2idx = {u: i for i, u in enumerate(neighbors)}
            idx2node = {i: u for i, u in enumerate(neighbors)}

            try:
                self._degree = self.degree[neighbors]
            except IndexError as err:
                print('Error:', err)
                print('neighbors:', neighbors)
                neighbors = target_candidates

            self._degree = self.degree[neighbors]
            self._adj = self.adj[neighbors][:, neighbors]
            self._adj_tilde = self.adj_tilde[neighbors][:, neighbors]
            self._adj_norm = self.adj_norm[neighbors][:, neighbors]
            self._X = self.X[neighbors]

            target_idx = node2idx[target_node]
            candidates_idx = [node2idx[u] for u in candidates]
            mask_idx = np.array([[node2idx[a], node2idx[b]] for a, b in mask if a in neighbors and b in neighbors])

            if self.verbose:
                print('degree:', self._degree)
                print('adj:', self._adj)
                print('adj_norm:', self._adj_norm)
        else:
            target_idx = target_node
            self._degree = np.copy(self.degree)
            self._adj = np.copy(self.adj)
            self._adj_tilde = np.copy(self.adj_tilde)
            self._adj_norm = np.copy(self.adj_norm)
            self._X = np.copy(self.X)
            mask_idx = mask
            neighbors = []

        t0 = time.time()
        XW = self._X @ self.W1
        # create two variables for the perturbation (positive changes and negative changes)
        epsilon_plus = cp.Variable(shape=self._adj.shape, name='epsilon_plus', symmetric=True)
        epsilon_minus = cp.Variable(shape=self._adj.shape, name='epsilon_minus', symmetric=True)

        # convex variable for the adjacency matrix, and adj_norm with target node
        # adj_norm_cvx = self._adj_norm + epsilon_plus
        adj_norm_cvx = self._adj_norm - epsilon_minus + epsilon_plus
        # adj_norm_v = self._adj_norm[target_idx] + epsilon_plus[target_idx]
        adj_norm_v = self._adj_norm[target_idx] - epsilon_minus[target_idx] + epsilon_plus[target_idx]

        if self.bias:
            H2_hat_cvx = adj_norm_cvx @ XW + np.tile(self.b1, [adj_norm_cvx.shape[0], 1])
            H2_hat = self._adj_norm @ XW + self.b1
        else:
            H2_hat_cvx = adj_norm_cvx @ XW
            H2_hat = self._adj_norm @ XW

        if self.deletion:
            H2_hat_cvx = H2_hat_cvx @ self.del1
            H2_hat = H2_hat @ self.del1


        H2 = relu(H2_hat)
        H2_cvx = cp.Variable(shape=(H2_hat.shape[0], H2_hat.shape[1]), name="H2", nonneg=True)
        H2_cvx.value = H2 

        c = self._compute_c(correct_label, target_label)
        try:
            optimization_problem = self._build_optimization_problem(
                c, target_idx, candidates_idx, adj_norm_v, adj_norm_cvx, 
                epsilon_minus, epsilon_plus, XW, H2_hat_cvx, H2_cvx,
                mask=mask_idx
            )
            # optimization_problem = self._build_optimization_problem(c, target_idx, candidates_idx, adj_norm_v, adj_norm_cvx, epsilon_minus, epsilon_plus, XW, H2_hat_cvx, H2_cvx)
        except ValueError as e:
            raise e
        except AssertionError as e:
            raise e
        prob_built_time = time.time() - t0

        delta = 0.
        try:
            if self.num_layer == 1:
                prob_state = optimization_problem.solve(solver=cp.CPLEX, verbose=self.verbose)
                if prob_state == 'infeasible':
                    return {'fragile': False, 'error': True, 'built_time': prob_built_time}
                # print('opt_value:', optimization_problem.value, ', b:', (c.T @ self.b1)[0], ', c', c)
                opt_value = optimization_problem.value + (c.T @ self.b1)[0] if self.bias else 0
                fragile = True if opt_value < 0 else False
                # print('opt_value:', opt_value)
                try:
                    vars_opt = {v.name(): v.value for v in optimization_problem.variables()}
                    best_adj_pert = self._adj_norm - vars_opt['epsilon_minus'] + vars_opt['epsilon_plus']
                except TypeError as err:
                    return {'fragile': False, 'error': True, 'built_time': prob_built_time}

                _mask = np.where(self._adj == 1, np.zeros_like(self._adj), np.ones_like(self._adj))
                potential_perturbations = np.multiply(_mask, np.triu(np.abs(self._adj_norm - best_adj_pert), 1))
                # Pick top 5 as the perturbations
                sorted_idx = np.argsort(potential_perturbations, axis=None)[::-1][:self.args.candidate_size]
                pert_indices = np.unravel_index(sorted_idx, potential_perturbations.shape)
                # pert_indices = np.where(potential_perturbations > 0.1)
                adj_diff = potential_perturbations[pert_indices]
                perturbations = np.concatenate(pert_indices).reshape(2, -1).T

                if candidates is not None: # map back to the original node index
                    perturbations = np.array([(idx2node[i], idx2node[j]) for i, j in perturbations])
                else:
                    raise NotImplementedError('Not implemented for the case of one-hop nodes.')
                
                return {
                    'fragile': True if fragile else False,
                    'perturbations': perturbations.tolist(),
                    'error': False,
                    'buit_time': prob_built_time,
                }

            elif self.num_layer == 3:
                prob_state = optimization_problem.solve(solver=solver)
                if prob_state == 'infeasible':
                    # print(f'Infeasible problem at target node {target_node}!')
                    return {'fragile': False, 'error': True}

                problems = [optimization_problem]
                best_uppers = [optimization_problem.upper_bound + (c.T @ self.b3)[0]]
                best_lowers = [optimization_problem.value + (c.T @ self.b3)[0]]
                solve_times = [optimization_problem.problem.solver_stats.solve_time]

                worst_lower = best_lowers[0]
                logit_diff_before = gcn_forward(self.adj_norm, self.X, self.theta, target_node, c).squeeze().item()
                # print('logit_diff_before:', logit_diff_before)
                # if logit_diff_before < 0:
                #     breakpoint()

                # robust = False
                fragile = False
                to_branch = optimization_problem
                for step in range(int(self.max_iter)):
                    lower_bounds, upper_bounds = bounds(problems)
                    if self.bias:
                        best_upper_bound = np.min(upper_bounds) + (c.T @ self.b3)[0]
                    else:
                        best_upper_bound = np.min(upper_bounds)
                    # print('at step:', step, ', best_uppper_bound:', best_upper_bound, ', worst_lower:', worst_lower)
                    if step > 0:
                        best_uppers.append(best_upper_bound)

                    if best_upper_bound + delta < 0:
                        fragile = True
                        # print('break from best_upper_bound < 0!')
                        break

                    if self.bias:
                        problems = purge(problems, best_upper_bound, constant=(c.T @ self.b3)[0])
                    else:
                        problems = purge(problems, best_upper_bound)

                    open_problems = [p for p in problems if p.open]
                    if len(open_problems) == 0:
                        fragile = best_upper_bound + delta < 0
                        # print('break from len(open_problems) == 0!')
                        break

                    if self.bias:
                        worst_lower_n = np.min([p.value for p in open_problems]) + (c.T @ self.b3)[0]
                    else:
                        worst_lower_n = np.min([p.value for p in open_problems])

                    # assert worst_lower_n - worst_lower >= -1e-10
                    worst_lower = worst_lower_n
                    if step > 0:
                        best_lowers.append(worst_lower)

                    if worst_lower > 0:
                        fragile = False
                        # print('break from worst_lower > 0!')
                        break

                    if best_upper_bound - worst_lower < self.tolerance:
                        fragile = False
                        # print('break from best_upper_bound - worst_lower < self.tolerance!')
                        break

                    to_branch = open_problems[np.argmin([p.value for p in open_problems])]
                    to_branch.branch()
                    for child in to_branch.children:
                        child.solve(solver=solver)
                        solve_times.append(child.problem.solver_stats.solve_time)
                        problems.append(child)

                best_prob = to_branch
                try:
                    best_adj_pert = self._adj_norm - best_prob.vars_opt['epsilon_minus'] + best_prob.vars_opt['epsilon_plus']
                except TypeError as err:
                    return {'fragile': False, 'error': True, 'built_time': prob_built_time}
            else:

                prob_state = optimization_problem.solve(solver=solver)
                if prob_state == 'infeasible':
                    # print(f'Infeasible problem at target node {target_node}!')
                    return {'fragile': False, 'error': True}

                problems = [optimization_problem]
                if self.bias:
                    best_uppers = [optimization_problem.upper_bound + (c.T @ self.b2)[0]]
                    best_lowers = [optimization_problem.value + (c.T @ self.b2)[0]]
                else:
                    best_uppers = [optimization_problem.upper_bound]
                    best_lowers = [optimization_problem.value]
                solve_times = [optimization_problem.problem.solver_stats.solve_time]

                worst_lower = best_lowers[0]
                logit_diff_before = gcn_forward(self.adj_norm, self.X, self.theta, target_node, c).squeeze().item()
                # print('logit_diff_before:', logit_diff_before)
                # if logit_diff_before < 0:
                #     breakpoint()

                # robust = False
                fragile = False
                to_branch = optimization_problem
                for step in range(int(self.max_iter)):
                    lower_bounds, upper_bounds = bounds(problems)
                    if self.bias:
                        best_upper_bound = np.min(upper_bounds) + (c.T @ self.b2)[0]
                    else:
                        best_upper_bound = np.min(upper_bounds)
                    if self.deletion:
                        best_upper_bound = best_upper_bound @ self.del2
                    # print('at step:', step, ', best_uppper_bound:', best_upper_bound, ', worst_lower:', worst_lower)
                    if step > 0:
                        best_uppers.append(best_upper_bound)

                    if best_upper_bound + delta < 0:
                        fragile = True
                        # print('break from best_upper_bound < 0!')
                        break

                    if self.bias:
                        problems = purge(problems, best_upper_bound, constant=(c.T @ self.b2)[0])
                    else:
                        problems = purge(problems, best_upper_bound)

                    open_problems = [p for p in problems if p.open]
                    if len(open_problems) == 0:
                        fragile = best_upper_bound + delta < 0
                        # print('break from len(open_problems) == 0!')
                        break

                    if self.bias:
                        worst_lower_n = np.min([p.value for p in open_problems]) + (c.T @ self.b2)[0]
                    else:
                        worst_lower_n = np.min([p.value for p in open_problems])

                    # assert worst_lower_n - worst_lower >= -1e-10
                    worst_lower = worst_lower_n
                    if step > 0:
                        best_lowers.append(worst_lower)

                    if worst_lower > 0:
                        fragile = False
                        # print('break from worst_lower > 0!')
                        break

                    if best_upper_bound - worst_lower < self.tolerance:
                        fragile = False
                        # print('break from best_upper_bound - worst_lower < self.tolerance!')
                        break

                    to_branch = open_problems[np.argmin([p.value for p in open_problems])]
                    to_branch.branch()
                    for child in to_branch.children:
                        child.solve(solver=solver)
                        solve_times.append(child.problem.solver_stats.solve_time)
                        problems.append(child)

                best_prob = to_branch
                try:
                    best_adj_pert = self._adj_norm - best_prob.vars_opt['epsilon_minus'] + best_prob.vars_opt['epsilon_plus']
                except TypeError as err:
                    return {'fragile': False, 'error': True, 'built_time': prob_built_time}

            # best_adj_pert - self._adj_norm
            _mask = np.where(self._adj == 1, np.zeros_like(self._adj), np.ones_like(self._adj))
            potential_perturbations = np.multiply(_mask, np.triu(np.abs(self._adj_norm - best_adj_pert), 1))

            # Pick top 5 as the perturbations
            sorted_idx = np.argsort(potential_perturbations, axis=None)[::-1][:int(self.args.candidate_size * self.data.num_nodes)]
            pert_indices = np.unravel_index(sorted_idx, potential_perturbations.shape)
            # pert_indices = np.where(potential_perturbations > 0.1)
            adj_diff = potential_perturbations[pert_indices]
            perturbations = np.concatenate(pert_indices).reshape(2, -1).T
            if perturbations[0][0] == perturbations[0][1] and fragile:
                # print('!!!', perturbations)
                fragile = False

            if candidates is not None: # map back to the original node index
                perturbations = np.array([(idx2node[i], idx2node[j]) for i, j in perturbations])
            
            # if fragile:
            #     print('best upper bound:', best_upper_bound)
            #     print('logit_diff_before:', logit_diff_before)
            #     epsilon_plus = best_prob.vars_opt['epsilon_plus']
            #     print('1P:', perturbations[0], ', epsilon plus:', epsilon_plus[np.nonzero(epsilon_plus)], ', position:', np.nonzero(epsilon_plus))
            #     print('epsilon_minus:', best_prob.vars_opt['epsilon_minus'][np.nonzero(best_prob.vars_opt['epsilon_minus'])], ', position:', np.nonzero(best_prob.vars_opt['epsilon_minus']))

            #     delta_deg_mx_i = (self._degree + 1)[:, np.newaxis].repeat(self._adj.shape[0], axis=1)
            #     delta_deg_mx_j = (self._degree + 1)[np.newaxis, :].repeat(self._adj.shape[0], axis=0)
            #     delta_adj_norm = 1 / np.sqrt(delta_deg_mx_i) / np.sqrt(delta_deg_mx_j)
            #     delta_adj_norm = np.where(self._adj_norm != 0, delta_adj_norm, 0)
            #     print('self._adj_norm - delta_adj_norm', (self._adj_norm - delta_adj_norm)[np.nonzero(self._adj_norm)])
            #     print('non zero:', np.nonzero(self._adj_norm))
            #     print(self._degree)
            # print(f'target node: {target_node}, target label: {target_label}, twohop nodes: {twohops}, perturbations: {perturbations}')

            if self.verbose:
                print('fragile:', fragile)
                print('best_upper_bound:', best_upper_bound)
                # print('adj_norm:', best_adj_pert)
                print('perturbations:', perturbations)
                print('adj_diff:', adj_diff)

            return {
                'fragile': True if fragile else False,
                'fragile_score': best_upper_bound.item(),
                'best_uppers': best_uppers,
                'best_lowers': best_lowers, 
                'adj_pert': best_adj_pert,
                # 'best_perturbation': best_prob.vars_opt['epsilon_minus'],
                'perturbations': perturbations.tolist(),
                'adj_diff': adj_diff.tolist(),
                'logit_diff_before': logit_diff_before,
                'solve_times': solve_times,
                'neighbors': neighbors,
                'error': False,
                'buit_time': prob_built_time,
                'step': step,
            }

        except cp.error.SolverError as err:
            print('SolverError:', err)
            return {'fragile': False, 'error': True, 'built_time': prob_built_time}


    def _compute_c(self, correct_label, target_label):
        c = np.zeros(self.data.num_classes)
        c[correct_label] = 1
        c[target_label] = -1
        c = c[:, None]
        return c
    
    def _build_optimization_problem(
            self, c, target_idx, candidates_idx, 
            adj_norm_v, adj_cvx, epsilon_minus, epsilon_plus, 
            XW, H2_hat_cvx, H2_cvx,
            mask=None
        ):
    # def _build_optimization_problem(
    #         self, c, target_idx, 
    #         adj_norm_v, adj_cvx, 
    #         epsilon_minus, 
    #         epsilon_plus, 
    #         XW, H2_hat_cvx, H2_cvx,
    #         mask=None
    #     ):
        # domain_constraints, bounds = self._build_domin_constraints(target_idx, candidates_idx, adj_norm_v, adj_cvx, epsilon_minus, epsilon_plus)
        domain_constraints, bounds = self._build_domin_constraints(target_idx, adj_norm_v, adj_cvx, epsilon_minus, epsilon_plus)
        # row_lower, row_upper, row_l1_distance_upper, global_lower, global_upper, pos_lower, pos_upper = bounds

        # element_lower, element_upper = self._element_bounds()
        element_lower, element_upper = self._element_bounds_new(target_idx, candidates_idx)
        if self.verbose:
            print('element_lower:', element_lower)
            print('element_upper:', element_upper)
 
        epsilon_plus_upper_bound = element_upper - self._adj_norm
        epsilon_plus_upper_bound = np.where(epsilon_plus_upper_bound <= 0, 0, epsilon_plus_upper_bound)

        assert np.all(epsilon_plus_upper_bound >= 0), f'epsilon_plus_upper_bound < 0, {epsilon_plus_upper_bound[epsilon_plus_upper_bound < 0]}'
        
        delta_deg_mx_i = (self._degree + 1)[:, np.newaxis].repeat(self._adj.shape[0], axis=1)
        delta_deg_mx_j = (self._degree + 1)[np.newaxis, :].repeat(self._adj.shape[0], axis=0)
        delta_adj_norm = 1 / np.sqrt(delta_deg_mx_i) / np.sqrt(delta_deg_mx_j)
        delta_adj_norm = np.where(self._adj_norm != 0, delta_adj_norm, 0)
        element_constraints = [
            adj_cvx >= element_lower,
            adj_cvx <= element_upper,
            # epsilon_minus <= self._adj_norm - delta_adj_norm,
            epsilon_plus <= epsilon_plus_upper_bound,
        ]
        domain_constraints.extend(element_constraints)
 
        attack_label_pos_mask = np.ones_like(self._adj)
        # attack_nodes = self.data.train_set.nodes[self.data.train_set.y == self.target_label].tolist()
        # attack_nodes = list(set(attack_nodes) - set(self.data.adj_list[target_idx]))
        attack_label_pos_mask[target_idx, candidates_idx] = 0
        attack_label_pos_mask[candidates_idx, target_idx] = 0
        attack_label_constraints = [
            cp.multiply(attack_label_pos_mask, epsilon_plus) == 0,
        ]
        domain_constraints.extend(attack_label_constraints)

        # assert len(candidates_idx) == 1, 'candidates_idx should be of length 1.'
        # attack_label_neg_mask = np.ones_like(self._adj)
        # attack_label_neg_mask[target_idx, np.nonzero(self._adj_tilde[target_idx])[0]] = 0
        # attack_label_neg_mask[np.nonzero(self._adj_tilde[target_idx])[0], target_idx] = 0
        # attack_label_neg_mask[candidates_idx[0], np.nonzero(self._adj_tilde[candidates_idx[0]])[0]] = 0
        # attack_label_neg_mask[np.nonzero(self._adj_tilde[candidates_idx[0]])[0], candidates_idx[0]] = 0
        # attack_label_neg_constraints = [
        #     cp.multiply(attack_label_neg_mask, epsilon_minus) == 0,
        # ]
        # domain_constraints.extend(attack_label_neg_constraints)

        # mask the perturbations that are found
        # if mask is not None and mask.shape[0] > 0:
        #     print('!!!!', 'mask:', mask)
        #     adj_mask = np.zeros_like(self._adj)
        #     adj_mask[mask.T[0], mask.T[1]] = 1
        #     adj_mask[mask.T[1], mask.T[0]] = 1
        #     mask_constraints = [
        #         cp.multiply(adj_mask, epsilon_minus) == 0,
        #         cp.multiply(adj_mask, epsilon_plus) == 0,
        #     ]
        #     domain_constraints.extend(mask_constraints)


        if self.num_layer == 1:
            WC = self.W1 @ c
            self.x_var = adj_norm_v
            AX = self.x_var @ self._X

            prb = cp.Problem(cp.Minimize(AX @ WC), domain_constraints)


            # prb = Problem(element_lower[target_idx], element_upper[target_idx], None, None, adj_norm_v, H2_cvx[target_idx], domain_constraints, x_orig=self._adj_norm[target_idx],
            #             eps_minus_x=epsilon_minus[target_idx],
            #             eps_plus_x=epsilon_plus[target_idx])
        elif self.num_layer == 3:
            
            # H2 = relu(H2_hat)
            # H2_cvx = cp.Variable(shape=(H2_hat.shape[0], H2_hat.shape[1]), name="H2", nonneg=True)
            # H2_cvx.value = H2 
            I2, I2_minus, I2_plus, R2, R2_I, S2, S2_I = self._relu_triangle_relaxation(XW, element_lower, element_upper)
            
            H2_cvx_I = cp.multiply(I2, H2_cvx)
            H2_cvx_I_plus = cp.multiply(I2_plus, H2_cvx)
            H2_cvx_I_minus = cp.multiply(I2_minus, H2_cvx)
            H2_hat_cvx_I = cp.multiply(I2, H2_hat_cvx)
            H2_hat_cvx_I_plus = cp.multiply(I2_plus, H2_hat_cvx)
            triangle_constraints = [
                H2_cvx >= H2_hat_cvx,
                H2_cvx_I_minus == 0,
                H2_cvx_I_plus == H2_hat_cvx_I_plus,
                cp.multiply(S2_I - R2_I, H2_cvx_I) <= cp.multiply(S2_I, H2_hat_cvx_I - R2_I)
            ]

            H2 = copy.deepcopy(np.array(H2_cvx.value))
            H2W = H2 @ self.W2
            H3_hat_cvx = adj_cvx @ H2W
            H3_hat = self._adj_norm @ H2W
            if self.bias:
                H3_hat_cvx = H3_hat_cvx + np.tile(self.b2, [H3_hat_cvx.shape[0], 1])
                H3_hat = H3_hat + np.tile(self.b2, [H3_hat.shape[0], 1])
            
            H3 = relu(H3_hat)
            H3_cvx = cp.Variable(shape=(H3_hat.shape[0], H3_hat.shape[1]), name="H3", nonneg=True)
            H3_cvx.value = H3 

            I3, I3_minus, I3_plus, R, R3_I, S, S3_I = self._relu_triangle_relaxation_layer3(H2W, element_lower, element_upper)
            H3_cvx_I = cp.multiply(I3, H3_cvx)
            H3_cvx_I_plus = cp.multiply(I3_plus, H3_cvx)
            H3_cvx_I_minus = cp.multiply(I3_minus, H3_cvx)
            H3_hat_cvx_I = cp.multiply(I3, H3_hat_cvx)
            H3_hat_cvx_I_plus = cp.multiply(I3_plus, H3_hat_cvx)
            triangle_constraints.extend([
                H3_cvx >= H3_hat_cvx,
                H3_cvx_I_minus == 0,
                H3_cvx_I_plus == H3_hat_cvx_I_plus,
                cp.multiply(S3_I - R3_I, H3_cvx_I) <= cp.multiply(S3_I, H3_hat_cvx_I - R3_I)
            ])

            L_x = element_lower[target_idx]
            U_x = element_upper[target_idx]
            L_y = np.maximum(R, 0)
            U_y = np.maximum(S, 0)
            WC = self.W3 @ c

            x_var = adj_norm_v
            z_lower = (L_y @ relu(WC) - U_y @ relu(-WC))[:, 0]
            z_upper = (U_y @ relu(WC) - L_y @ relu(-WC))[:, 0]

            assert np.all(z_lower <= z_upper), f'z_lower < z_upper, {z_lower[z_lower > z_upper]} < {z_upper[z_lower > z_upper]}'
            z = cp.Variable(z_lower.shape, name='z')
            z_constraints = [
                z >= z_lower,
                z <= z_upper,
                z == (H3_cvx @ WC)[:, 0]
            ]
            constraints = {
                'domain_constraints': domain_constraints,
                'triangle_constraints': triangle_constraints,
                'z_constraints': z_constraints
            }

            extra_var_eps_minus = cp.Variable(epsilon_minus[target_idx].shape,
                                            'extra_var_eps_minus')
            extra_var_eps_plus = cp.Variable(epsilon_plus[target_idx].shape,
                                            'extra_var_eps_plus')
            constraints['extra_constraints'] = [
                extra_var_eps_minus == epsilon_minus[target_idx],
                extra_var_eps_plus == epsilon_plus[target_idx],
            ]

            prb = Problem(L_x, U_x, z_lower, z_upper, x_var, z, constraints,
                        x_orig=self._adj_norm[target_idx],
                        eps_minus_x=extra_var_eps_minus,
                        eps_plus_x=extra_var_eps_plus)

        else:
            triangle_constraints, R, S = self._build_triangle_constraints(XW, element_lower, element_upper, H2_hat_cvx, H2_cvx)

            L_x = element_lower[target_idx]
            U_x = element_upper[target_idx]
            L_y = np.maximum(R, 0)
            U_y = np.maximum(S, 0)
            WC = self.W2 @ c

            x_var = adj_norm_v

            z_lower = (L_y @ relu(WC) - U_y @ relu(-WC))[:, 0]
            z_upper = (U_y @ relu(WC) - L_y @ relu(-WC))[:, 0]

            assert np.all(z_lower <= z_upper), f'z_lower < z_upper, {z_lower[z_lower > z_upper]} < {z_upper[z_lower > z_upper]}'
            z = cp.Variable(z_lower.shape, name='z')
            z_constraints = [
                z >= z_lower,
                z <= z_upper,
                z == (H2_cvx @ WC)[:, 0]
            ]
            constraints = {
                'domain_constraints': domain_constraints,
                'triangle_constraints': triangle_constraints,
                'z_constraints': z_constraints
            }

            extra_var_eps_minus = cp.Variable(epsilon_minus[target_idx].shape,
                                            'extra_var_eps_minus')
            extra_var_eps_plus = cp.Variable(epsilon_plus[target_idx].shape,
                                            'extra_var_eps_plus')
            constraints['extra_constraints'] = [
                extra_var_eps_minus == epsilon_minus[target_idx],
                extra_var_eps_plus == epsilon_plus[target_idx],
            ]

            prb = Problem(L_x, U_x, z_lower, z_upper, x_var, z, constraints,
                        x_orig=self._adj_norm[target_idx],
                        eps_minus_x=extra_var_eps_minus,
                        eps_plus_x=extra_var_eps_plus)

        return prb
        # element_lower, element_upper, row_lower, row_upper, row_l1_distance_upper, global_lower, global_upper, pos_lower, pos_upper


    # def _build_domin_constraints(self, target_idx, candidates_idx, adj_v_cvx, adj_cvx, epsilon_minus, epsilon_plus):
    def _build_domin_constraints(self, target_idx, adj_v_cvx, adj_cvx, epsilon_minus, epsilon_plus):
        # row_lower, row_upper, row_l1_distance_upper = self._row_bounds_new(target_idx, candidates_idx)
        row_lower, row_upper, row_l1_distance_upper = self._row_bounds(target_idx)
        global_lower, global_upper = self._global_bound(target_idx)
        # assert global_lower <= global_upper, f'global_lower > global_upper, {global_lower} > {global_upper}'
        pos_lower, pos_upper = self._pos_bounds(target_idx)
        assert pos_lower <= pos_upper, f'pos_lower > pos_upper, {pos_lower} > {pos_upper}'
        if self.verbose:
            print('row_lower:', row_lower)
            print('row_upper:', row_upper)
 
        domain_constraints = [
            cp.diag(epsilon_plus) == 0,
            epsilon_minus >= 0,
            epsilon_plus >= 0,

            cp.sum(epsilon_plus) >= pos_lower * 2,
            cp.sum(epsilon_plus) <= pos_upper * 2,

            # tie together the row of node v with the matrix variable
            adj_v_cvx == adj_cvx[target_idx],

            # row-wise bounds
            cp.sum(adj_cvx, axis=1) >= row_lower,
            cp.sum(adj_cvx, axis=1) <= row_upper,

            # row-wise bounds II
            # cp.sum(cp.abs(adj_cvx - self._adj_norm), axis=1) <= row_l1_distance_upper,
            
            # global upper bound
            cp.sum(cp.abs(adj_cvx - self._adj_norm)) <= global_upper * 2,
        ]
        return domain_constraints, (row_lower, row_upper, row_l1_distance_upper, None, global_upper, pos_lower, pos_upper)
    
    def _get_area(self, i, j, target_idx, candidates_idx):
        if i == target_idx and (j == target_idx or j in candidates_idx):
            return 1
        elif i == target_idx and j not in candidates_idx:
            return 2
        elif i in candidates_idx:
            return 3
        elif i not in candidates_idx and j not in candidates_idx:
            return 4

    
    def _element_bounds_new(self, target_idx, candidates_idx):
        """ we assume that there is exactly one perturbation between the target node and candidate nodes
        """
        # print('target_idx:', target_idx)
        # print('candidates_idx:', candidates_idx)

        upper = np.copy(self._adj_norm)
        lower = np.copy(self._adj_norm)

        target_neighbors = np.nonzero(self._adj[target_idx])[0]
        candidate_neighbors = np.nonzero(self._adj[candidates_idx[0]])[0]
        upper[target_idx, candidates_idx[0]] = 1 / np.sqrt(self._degree[target_idx] + 1) / np.sqrt(self._degree[candidates_idx[0]] + 1)
        upper[candidates_idx[0], target_idx] = 1 / np.sqrt(self._degree[target_idx] + 1) / np.sqrt(self._degree[candidates_idx[0]] + 1)

        lower[target_idx, target_idx] = 1 / (self._degree[target_idx] + 1)
        lower[candidates_idx[0], candidates_idx[0]] = 1 / (self._degree[candidates_idx[0]] + 1)
        lower[target_idx, target_neighbors] = 1 / np.sqrt(self._degree[target_idx] + 1) / np.sqrt(self._degree[target_neighbors])
        lower[target_neighbors, target_idx] = 1 / np.sqrt(self._degree[target_idx] + 1) / np.sqrt(self._degree[target_neighbors])
        lower[candidates_idx[0], candidate_neighbors] = 1 / np.sqrt(self._degree[candidates_idx[0]] + 1) / np.sqrt(self._degree[candidate_neighbors])
        lower[candidate_neighbors, candidates_idx[0]] = 1 / np.sqrt(self._degree[candidates_idx[0]] + 1) / np.sqrt(self._degree[candidate_neighbors])

        # upper = np.zeros_like(self._adj)
        # lower = np.zeros_like(self._adj)
        # for i in range(self._adj.shape[0]):
        #     for j in range(i, self._adj.shape[0]):
        #         area = self._get_area(i, j, target_idx, candidates_idx)
        #         # print(i, j, ', area:', area)
        #         if area == 1:
        #             if i == j: # diagonal
        #                 lower[i, j] = 1 / (self._degree[i] + 1)
        #                 upper[i, j] = 1 / (self._degree[i])
        #             else:
        #                 lower[i, j] = 0
        #                 upper[i, j] = 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[j])
        #         elif area == 2:
        #             lower[i, j] = np.minimum(self._adj_norm[i, j], 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[j]))
        #             upper[i, j] = np.maximum(self._adj_norm[i, j], 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[j]))
        #             # upper[i, j] = self._adj_norm[i, j]
        #         elif area == 3:
        #             if i == j:
        #                 lower[i, j] = 1 / (self._degree[i] + 1)
        #                 upper[i, j] = self._adj_norm[i, j]
        #                 # print(f'check {i}-{j}', upper[i, j], self._adj_norm[i, j])
        #                 # if upper[i, j] != self._adj_norm[i, j]:
        #                 #     print(f'check {i}-{j}', upper[i, j], self._adj_norm[i, j])
        #                 #     exit(0)
        #             else:
        #                 lower[i, j] = np.minimum(self._adj_norm[i, j], 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[j] + 1))
        #                 upper[i, j] = np.maximum(self._adj_norm[i, j], 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[j] + 1))
        #         elif area == 4:
        #             lower[i, j] = self._adj_norm[i, j]
        #             upper[i, j] = self._adj_norm[i, j]

        #         if upper[i, j] - self._adj_norm[i, j] < -1e-5:
        #             print(f'WARNING: upper bound is smaller than the original value, at {i}-{j}. {upper[i, j]} < {self._adj_norm[i, j]}')
        #             upper[i, j] = self._adj_norm[i, j]

        # lower = lower + lower.T - np.diag(np.diag(lower))
        # upper = upper + upper.T - np.diag(np.diag(upper))

        # lower = np.where(lower == 0, np.zeros_like(lower) + 1e-5, lower)
        # upper = np.where(upper == 0, np.zeros_like(upper) + 1e-5, upper)

        assert np.all(lower <= upper), f'Invalid element bounds, lower is greater than upper. {lower[lower>upper]} > {upper[lower>upper]}.' 
        return lower, upper

 
    def _element_bounds(self):
        # Case 1: 1 / sqrt(d_u + 1) + 1 / sqrt(d_v + 1)
        # delta_deg_mx_i = (self._degree + 1).reshape(-1, 1).repeat(1, self._adj.shape[0])
        # delta_deg_mx_j = (self._degree + 1).repeat(self._adj.shape[0], 1)
        if self._degree.shape[0] != self._adj.shape[0]:
            breakpoint()
        delta_deg_mx_i = (self._degree + 1)[:, np.newaxis].repeat(self._adj.shape[0], axis=1)
        delta_deg_mx_j = (self._degree + 1)[np.newaxis, :].repeat(self._adj.shape[0], axis=0)
        delta_adj_norm = 1 / np.sqrt(delta_deg_mx_i) / np.sqrt(delta_deg_mx_j)

        upper = np.maximum(self._adj_norm, delta_adj_norm)
        lower = np.where(self._adj == 0, self._adj, delta_adj_norm)

        assert np.all(lower <= upper), f'Invalid element bounds, lower is greater than upper. {lower[lower>upper]} > {upper[lower>upper]}.'
        return lower, upper
    
    def _row_bounds_new(self, target_idx, candidates_idx):
        sorted_indices = np.argsort(self._degree)
        # sorted_indices = np.argsort(self._degree[candidates_idx])
        smallest_deg_candidate = candidates_idx[sorted_indices[0]]
        largest_deg_candidate = candidates_idx[sorted_indices[-1]]

        deg_v = self._degree[target_idx]

        lower = np.zeros(self._adj.shape[0])
        upper = np.zeros(self._adj.shape[0])
        l1_distance_upper = np.zeros(self._adj.shape[0])
        for i in range(self._adj.shape[0]):
            _neighbors = self._adj[i].nonzero()[0]
            case1_low = np.sum(1/np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[_neighbors])) \
                    + 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[largest_deg_candidate] + 1) \
                    + 1 / (self._degree[i] + 1)  
            case1_high = np.sum(1/np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[_neighbors])) \
                    + 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[smallest_deg_candidate] + 1) \
                    + 1 / (self._degree[i] + 1)  
            if i == target_idx:
                lower[i] = np.sum(1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[_neighbors])) \
                        + 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[largest_deg_candidate] + 1) \
                        + 1 / (self._degree[i] + 1)
                upper[i] = np.sum(1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[_neighbors])) \
                        + 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[smallest_deg_candidate] + 1) \
                        + 1 / (self._degree[i] + 1)
                l1_distance_upper[i] = np.sum((1 / np.sqrt(self._degree[i]) - 1 / np.sqrt(self._degree[i] + 1)) / np.sqrt(self._degree[_neighbors])) \
                                + 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[smallest_deg_candidate] + 1)
            elif i in candidates_idx:
                # low[i] = np.sum(1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[_neighbors])) \
                #             + 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(largest_deg_candidate + 1) + 1 / (self._degree[i] + 1)
                lower[i] = np.minimum(self._adj_norm[i].sum(), case1_low)
                upper[i] = np.maximum(self._adj_norm[i].sum(), case1_high)
                
                # delta_row_sum = np.sum(1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[_neighbors])) \
                #             + 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(deg_v + 1) + 1 / (self._degree[i] + 1)
                # lower[i] = np.minimum(self._adj_norm[i].sum(), delta_row_sum)
                # upper[i] = np.maximum(self._adj_norm[i].sum(), delta_row_sum)

                l1_distance_upper[i] = np.sum((1 / np.sqrt(self._degree[i]) - 1 / np.sqrt(self._degree[i] + 1)) / np.sqrt(self._degree[_neighbors])) \
                                + 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(deg_v + 1)
            else:
                lower[i] = np.minimum(self._adj_norm[i].sum(), case1_low)
                upper[i] = np.maximum(self._adj_norm[i].sum(), case1_high)
                # _degree = self._degree.clone()
                # _degree[[target_idx] + candidates_idx] += 1 
                # lower[i] = np.sum(1 / np.sqrt(_degree[i]) / np.sqrt(_degree[_neighbors])) + 1 / self._degree[i]
                # upper[i] = self._adj_norm[i].sum()
                l1_distance_upper[i] = 1 /np.sqrt(self._degree[i]) / np.sqrt(deg_v) -  1 / np.sqrt(self._degree[i] + 1) / np.sqrt(deg_v + 1) \
                                + 1 / np.sqrt(self._degree[i]) / np.sqrt(self._degree[smallest_deg_candidate]) - 1 / np.sqrt(self._degree[i]) / np.sqrt(self._degree[smallest_deg_candidate] + 1)
        
        assert np.all(lower - upper < 1e5), f'Invalid row bounds, lower is greater than upper. {lower[lower>upper]} > {upper[lower>upper]}, at {np.where(lower > upper)}.'
        lower = np.where(lower > upper, upper, lower)

        return lower, upper, l1_distance_upper

    
    def _row_bounds(self, target_idx):
        """ Eqn (6) in the paper
        """
        # u_deg, v_deg = np.sort(self._degree).values[:2] # the two smallest degrees
        # upper = np.sum(self._adj_norm, dim=1) + 1 / np.sqrt(self._degree * (u_deg +1)) + 1 / np.sqrt(self._degree * (v_deg + 1))
        # # upper = upper.cpu().numpy()

        # # Case 1 & 2
        # lower = torch.zeros(self._adj.shape[0])
        # row_l1_distance_upper = torch.zeros(self._adj.shape[0])
        # for i in range(self._adj.shape[0]):
        #     neighbors = self._adj[i].nonzero()[0]

        #     _degree = np.copy(self._degree)
        #     _degree[neighbors] = 0 # filter out the neighbors
        #     u_deg = np.sort(_degree).values[-1] # the largest degree
            
        #     lower[i] = torch.sum(1 / torch.sqrt(self._degree[i] + 1) / torch.sqrt(self._degree[neighbors])) + (1 / torch.sqrt((self._degree[i] + 1) * (u_deg + 1)))
        #     row_l1_distance_upper[i] = 1 / torch.sqrt(self._degree[i]) / torch.sqrt(u_deg + 1) + 1 / torch.sqrt(self._degree[i]) / torch.sqrt(v_deg + 1)
        
        # return lower, upper, row_l1_distance_upper

        sorted_indices = np.argsort(self._degree)
        # sorted_indices = np.argsort(self._degree[candidates_idx])
        smallest_deg_candidate = list(range(self._adj.shape[0]))[sorted_indices[0]]
        largest_deg_candidate = list(range(self._adj.shape[0]))[sorted_indices[-1]]

        deg_v = self._degree[target_idx]

        lower = np.zeros(self._adj.shape[0])
        upper = np.zeros(self._adj.shape[0])
        l1_distance_upper = np.zeros(self._adj.shape[0])
        for i in range(self._adj.shape[0]):
            _neighbors = self._adj[i].nonzero()[0]
            case1_low = np.sum(1/np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[_neighbors])) \
                    + 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[largest_deg_candidate] + 1) \
                    + 1 / (self._degree[i] + 1)  
            case1_high = np.sum(1/np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[_neighbors])) \
                    + 1 / np.sqrt(self._degree[i] + 1) / np.sqrt(self._degree[smallest_deg_candidate] + 1) \
                    + 1 / (self._degree[i] + 1)

            lower[i] = np.minimum(self._adj_norm[i].sum(), case1_low)
            upper[i] = np.maximum(self._adj_norm[i].sum(), case1_high)
        return lower, upper, l1_distance_upper
 
    def _global_bound(self, target_idx):
        """ Eqn (9) in the paper
        """
        d_a = self._degree[target_idx]
        idx_max = np.argmax(self._degree)
        idx_min = np.argmin(self._degree)
        d_max = self._degree[idx_max]
        d_min = self._degree[idx_min]
        # idx_max = np.argmax(self._degree[candidates_idx])
        # idx_min = np.argmin(self._degree[candidates_idx])
        # d_max = self._degree[candidates_idx][idx_max]
        # d_min = self._degree[candidates_idx][idx_min]
        # print('d:', d_a, d_max, d_min)
        # print('idx:', idx_max, idx_min)

        _delta_adj_norm_max = np.copy(self._adj_norm[idx_max])
        _delta_adj_norm_max[idx_max] = 0
        lower = 4 * (1 / d_max - 1 / (d_max + 1)) + 2 * np.sum(_delta_adj_norm_max * (1 / np.sqrt(d_max) - 1 / np.sqrt(d_max + 1)))
        _delta_adj_norm_min = np.copy(self._adj_norm[idx_min])
        _delta_adj_norm_min[idx_min] = 0
        upper = 4 * (1 / d_min - 1 / (d_min + 1)) + 2 * np.sum(_delta_adj_norm_min * (1 / np.sqrt(d_min) - 1 / np.sqrt(d_min + 1))) 

        return lower, upper
    
    def _pos_bounds(self, target_idx):
        d_a = self._degree[target_idx]
        d_max = np.max(self._degree)
        d_min = np.min(self._degree)
        lower = 1 / np.sqrt(d_a + 1) / np.sqrt(d_max + 1)
        upper = 1 / np.sqrt(d_a + 1) / np.sqrt(d_min + 1)
        return lower, upper
        

    def _relu_triangle_relaxation(self, XW, element_lower, element_upper):
        H2_hat = self._adj_norm @ self._X @ self.W1
        if self.bias:
            H2_hat = H2_hat + self.b1

        XW_plus = np.maximum(XW, 0)
        XW_minus = -np.minimum(XW, 0)

        S = element_upper @ XW_plus - element_lower @ XW_minus
        R = element_lower @ XW_plus - element_upper @ XW_minus
        if self.bias:
            S = S + self.b1
            R = R + self.b1

        assert np.all(R <= S), f'R <= S, {R[R > S]} <= {S[R > S]}'

        I = (S > 0) & (R < 0)
        I_plus = (S >= 0) & (R >= 0)
        I_minus = (S <= 0) & (R <= 0)

        assert np.all(I + I_plus + I_minus)
        # print(H2_hat.shape, element_lower.shape, element_upper.shape, XW.shape, I.shape, I_plus.shape, I_minus.shape)
        assert (H2_hat - S <= 1e-5).all(), f'at {np.where(H2_hat - S > 1e-5)}, upper_bound {element_upper}, adj_norm: {self._adj_norm}.'
        assert (H2_hat - R >= -1e-5).all(), f'H2_hat: {H2_hat[H2_hat - R < -1e-5]} < R: {R[H2_hat - R < -1e-5]}, at {np.where(H2_hat - R < -1e-5)}'
        S_I = np.multiply(S, I)
        R_I = np.multiply(R, I)

        assert np.all(R_I <= S_I), f'R_I <= S_I, {R_I[R_I > S_I]} <= {S_I[R_I > S_I]}'
        return I, I_minus, I_plus, R, R_I, S, S_I
    
    def _relu_triangle_relaxation_layer3(self, H2W, element_lower, element_upper):
        H3_hat = self._adj_norm @ H2W
        if self.bias:
            H3_hat = H3_hat + self.b2
        
        H2W_plus = np.maximum(H2W, 0)
        H2W_minus = -np.minimum(H2W, 0)

        S = element_upper @ H2W_plus - element_lower @ H2W_minus
        R = element_lower @ H2W_plus - element_upper @ H2W_minus
        if self.bias:
            S = S + self.b2
            R = R + self.b2
        assert np.all(R <= S), f'R <= S, {R[R > S]} <= {S[R > S]}'
        
        I = (S > 0) & (R < 0)
        I_plus = (S >= 0) & (R >= 0)
        I_minus = (S <= 0) & (R <= 0)

        assert np.all(I + I_plus + I_minus)
        assert (H3_hat - S <= 1e-5).all(), f'at {np.where(H3_hat - S > 1e-5)}, upper_bound {element_upper}, adj_norm: {self._adj_norm}.'
        assert (H3_hat - R >= -1e-5).all(), f'H3_hat: {H3_hat[H3_hat - R < -1e-5]} < R: {R[H3_hat - R < -1e-5]}, at {np.where(H3_hat - R < -1e-5)}'

        S_I = np.multiply(S, I)
        R_I = np.multiply(R, I)

        assert np.all(R_I <= S_I), f'R_I <= S_I, {R_I[R_I > S_I]} <= {S_I[R_I > S_I]}'

        return I, I_minus, I_plus, R, R_I, S, S_I

    
    def _build_triangle_constraints(self, XW, element_lower, element_upper, H2_hat_cvx, H2_cvx):
        I, I_minus, I_plus, R, R_I, S, S_I = self._relu_triangle_relaxation(XW, element_lower, element_upper)
        
        H2_cvx_I = cp.multiply(I, H2_cvx)
        H2_cvx_I_plus = cp.multiply(I_plus, H2_cvx)
        H2_cvx_I_minus = cp.multiply(I_minus, H2_cvx)
        H2_hat_cvx_I = cp.multiply(I, H2_hat_cvx)
        H2_hat_cvx_I_plus = cp.multiply(I_plus, H2_hat_cvx)
        triangle_constraints = [
            H2_cvx >= H2_hat_cvx,
            H2_cvx_I_minus == 0,
            H2_cvx_I_plus == H2_hat_cvx_I_plus,
            cp.multiply(S_I - R_I, H2_cvx_I) <= cp.multiply(S_I, H2_hat_cvx_I - R_I)
        ]
        return triangle_constraints, R, S
                           
    def _deg_norm(self, i_deg, j_deg):
        return 1 / torch.sqrt(i_deg) / torch.sqrt(j_deg)

class OnePerturbationFragility(object):

    """ 
    A class to find 1-perturbation fragile node tokens 
    
    @Deprecated
    This class is deprecated because randomized smoothing is not a good way to find fragile nodes.
    
    """

    def __init__(self, args, model: torch.nn.Module, data, device):
        """
        model: GNN model, normally is a surrogate model.
        data: the dataset.
        device: the device to run the model.
        """
        self.args = args
        self.model = model.to(device)
        self.data = data
        self.num_classes = data.num_classes
        self.dim = data.num_features
        self.adj = data.adjacency_matrix().to_dense().to(device)
        self.fea = data.x.to(device)
        self.num_nodes = data.num_nodes
        self.device = device

        self.m = Bernoulli(torch.tensor([0.5], device=self.device))

    def _sample_noise(self, v, d):
        # m = Bernoulli(torch.tensor([p]), device=device)
        adj = self.adj.int().clone().detach().to(self.device)
        adj_noise = adj.clone().detach().to(self.device)

        counts = np.zeros(self.num_classes, dtype=int)

        for _ in range(d):
            # mask = m.sample(adj.shape[1]).squeeze(-1).int()
            rand_inputs = torch.zeros_like(adj[v], device=self.device).int()
            candidate = torch.where(adj[v] == 0)[0].tolist()
            idx = random.choice(candidate)
            rand_inputs[idx] = 1

            if self.args.cpf_epsilon: # With epsilon
                mask = self.m.sample(adj[idx].shape).squeeze(-1).int()
                adj_noise[idx] = adj[idx] * mask + rand_inputs * (1 - mask)
            else: # Avoid epsilon 
                adj_noise[v] = adj[v] + rand_inputs
            
            adj_noise[:, v] = adj_noise[v]
            
            if self.args.cpf_retrain: # For poisoning attack
                _data = copy.deepcopy(self.data)
                _data.update_edge_index_by_adj(adj_noise)
                tmp_model = GNN(self.args, self.dim, self.num_classes, fix_weight=not self.args.cpf_random)
                tmp_model.train(_data, device=self.device)
                pred = tmp_model.predict(_data, self.device, target_nodes=[v])[0]
            else: # For evasion attack
                adj_noise_norm = utils.normalize(torch.eye(self.num_nodes, device=self.device) + adj_noise)
                with torch.no_grad():
                    pred = self.model(self.fea, adj_noise_norm)[v].argmax(0).item()

            counts[pred] += 1

        return counts
    

    def certify_fragile(self, v: int, label, N: int, alpha: float):
        """
        Find the fragile nodes in the graph.
        :param model: the model to be used
        :param data: the data to be used
        :param alpha: the alpha value
        :param d: the d value
        :return: the list of fragile nodes
        """
        counts_selection = self._sample_noise(v, N)
        d_ca = counts_selection[label].item()
        p_a_bar = proportion_confint(d_ca, N, alpha=alpha, method='beta')[1]

        fragile_classes = {}
        for c in range(self.num_classes):
            if c == label:
                continue
            d_c = counts_selection[c]
            p_c_bar = proportion_confint(d_c, N, alpha=alpha, method='beta')[0]
            if p_c_bar >= p_a_bar:
                fragile_classes[c] = p_c_bar
        sorted_fragile_classes = {k: v for k, v in sorted(fragile_classes.items(), key=lambda item: item[1], reverse=True)}
        return len(fragile_classes) > 0, list(sorted_fragile_classes.keys())



if __name__ == '__main__':
    pass