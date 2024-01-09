# Implementation of the paper:
# Certifiable Robustness of Graph Convolutional Networks under Structure Perturbations
#
# by Daniel Zügner and Stephan Günnemann.
# Published at KDD'20, August 2020 (virtual event).
#
# Copyright (C) 2020
# Daniel Zügner
# Technical University of Munich

import torch
import numpy as np
import scipy.sparse as sp
import cvxpy as cvx
from robust_gcn_structure import _utils
relu = lambda x: torch.maximum(torch.zeros_like(x), x)


class GCNNodeCertification:
    def __init__(self, i_th, A, twohop_nbs, onehop_ixs_th, onehop_nbs, X, model,
                 local_changes=5, global_changes=20, A_tilde=None, device=torch.device('cpu')):
        """
        GNN certification base class.

        Parameters
        ----------
        i_th: int
            the index of the target node in the list of twohop neighbors
        A: sp.spmatrix
            The [N,N] adjacency matrix of the graph.
        twohop_nbs: np.array
            The indices of the twohop neighbors of the target node.
        onehop_ixs_th: np.array
            The indices of the one-hop neighbors of the target node in the list of twohop neighbors.
        onehop_nbs: np.array
            The indices of the onehop neighbors of the target node.
        X: np.array
            The [N, D] node feature matrix.
        weights: List of np.arrays
            List containing [W1, b1, W2, b2], i.e. the weights and biases of the first and second layer.
        local_changes: int
            The maximum number of allowed edge perturbations per individual node.
        global_changes: int
            The maximum number of allowed edge perturbations in the graph.
        A_tilde: sp.spmatrix (optional)
            The adjacency matrix with added self loops; if provided, it does not have to be re-computed, leading
            to a speedup.
        """

        self.local_changes = local_changes
        self.global_changes = global_changes

        self.A = A
        # self.An = utils.preprocess_adj(self.A).toarray()
        self.An = _utils.preprocess_adj(self.A)

        self.model = model
        self.device = device

        # self.W1, self.b1, self.W2, self.b2 = weights
        weights = [w.detach() for w in model.parameters()]
        if len(weights) == 2:
            self.bias = False
            self.W1, self.W2 = weights
        else:
            self.bias = True
            self.W1, self.b1, self.W2, self.b2 = weights
        self.K = self.W2.shape[1]

        # We need several lists of indices throughout the certification.
        # * twohop_nbs: the indices of the twohop neighbors in the full graph.
        # * onehop_nbs: the indices of the onehop neighbors in the full graph.
        # * onehop_ixs_th: the indices of the onehop neighbors within the twohop neighbors.
        #   That is, twohop_nbs[onehop_ixs_th] == onehop_nbs
        # * i_th: the index of the target nodes within the twohop neighbors, i.e. twohop_nbs[i_th] == i
        # * i_oh: the index of the target nodes within the onehop neighbors, i.e. onehop_nbs[i_oh] == i
        self.twohop_nbs = twohop_nbs
        self.onehop_nbs = onehop_nbs
        self.onehop_ixs_th = onehop_ixs_th
        self.i_th = i_th
        self.i_oh = (self.onehop_nbs == self.twohop_nbs[self.i_th]).long().argmax()

        # Slice the relevant parts from the respective input matrices.
        self.X_th = X[self.twohop_nbs]
        self.An_oh_th = extract_submatrix(self.An, self.onehop_nbs, self.twohop_nbs)
        self.An_th_th = extract_submatrix(self.An, self.twohop_nbs)
        self.A_th_th = extract_submatrix(self.A, self.twohop_nbs)
        
        if A_tilde is None:
            # A_tilde = self.A + sp.eye(self.A.shape[0])
            A_tilde = torch.eye(self.A.size(0), device=self.device) + self.A
        self.A_mask_th = extract_submatrix(A_tilde, self.twohop_nbs)

        # if self.bias:
        #     self.XW = self.X_th @ self.W1 + self.b1
        # else:
        #     self.XW = self.X_th @ self.W1
        self.XW = self.X_th @ self.W1

        self.degs = self.A.sum(0)
        # self.degs = self.A.sum(0).A1
        self.degs_tilde = self.degs + 1
        self.degs_oh_tilde = self.degs[self.onehop_nbs] + 1
        self.degs_th_tilde = self.degs[self.twohop_nbs] + 1

        """ Modified by Kun Wu, 2023-07-13
            We only want to find 1-perturbation nodes, therefore, we set the max_delete to 1

            # keep at least one edge per node
            self.max_delete = np.maximum(self.degs_tilde.astype("int") - 2, 0)
            self.max_delete = np.minimum(self.max_delete, self.local_changes)
        """
        self.max_delete = torch.maximum(self.degs_tilde.long() - 2, torch.zeros_like(self.degs_tilde))
        self.max_delete = torch.minimum(self.max_delete, torch.zeros_like(self.max_delete) + self.local_changes)
        # self.max_delete = 1
        # assert ((self.degs_tilde - self.max_delete) > 0).all()
        # assert (self.max_delete >= 0).all()
        # assert (self.max_delete <= local_changes).all()

        self.A_upper = self.naive_entrywise_upper_bounds()
        self.A_upper_th = self.A_upper.clone()
        # self.A_upper_th = self.A_upper.copy()
        self.A_upper = self.A_upper[self.onehop_ixs_th]
        assert (self.A_upper - self.An_oh_th > -1e-9).all()

        # naive entrywise lower bounds (all zeros except diagonal)
        self.A_lower_th = self.naive_entrywise_lower()
        self.A_lower = self.A_lower_th[self.onehop_ixs_th]

        self.epsilon_plus_var = cvx.Variable(shape=(self.An_oh_th.shape[1], self.An_oh_th.shape[1]), name="epsilon_plus",
                                             symmetric=True)
        self.epsilon_minus_var = cvx.Variable(shape=(self.An_oh_th.shape[1], self.An_oh_th.shape[1]), name="epsilon_minus",
                                              symmetric=True)
        self.epsilon_plus = self.epsilon_plus_var
        self.epsilon_minus = self.epsilon_minus_var
        self.epsilon_plus_th = cvx.multiply(self.A_mask_th.cpu(), self.epsilon_plus_var)
        self.epsilon_minus_th = cvx.multiply(self.A_th_th.float().cpu(), self.epsilon_minus)
        self.epsilon_minus = self.epsilon_minus_th[self.onehop_ixs_th.cpu()]
        self.epsilon_plus = self.epsilon_plus_th[self.onehop_ixs_th.cpu()]

        self.An_cvx = self.An_oh_th.cpu() - self.epsilon_minus + self.epsilon_plus
        self.An_cvx_orig = self.An_th_th.cpu() - self.epsilon_minus_th + self.epsilon_plus_th

        self.An_i_val = (self.An_oh_th[self.i_oh, self.onehop_ixs_th.cpu()].cpu() + self.epsilon_plus[self.i_oh, self.onehop_ixs_th.cpu()]
                         - self.epsilon_minus[self.i_oh, self.onehop_ixs_th.cpu()])

        if self.bias:
            self.H2_hat_cvx = self.An_cvx @ self.XW.cpu() + torch.tile(self.b1.cpu(), [self.An_cvx.shape[0], 1])
        else:
            self.H2_hat_cvx = self.An_cvx @ self.XW.cpu() 

        if self.bias:
            H2_hat = self.An_oh_th @ self.XW + self.b1
        else:
            H2_hat = self.An_oh_th @ self.XW
        H2 = relu(H2_hat.cpu())

        self.H2_cvx = cvx.Variable(shape=(H2_hat.shape[0], H2_hat.shape[1]), name="H2", nonneg=True)
        self.H2_cvx.value = H2.cpu().numpy()

    def weights(self):
        if self.bias:
            return [self.W1, self.b1, self.W2, self.b2]
        else:
            return [self.W1, self.W2]

    def row_upper_bounds(self):
        row_upper = torch.zeros_like(self.twohop_nbs, device=self.device) + 1e9
        # The upper bounds from Eq. (9) only work for the one-hop neighbors of the target node:
        row_upper_oh = torch.tensor([row_upper_bound(ix, self.An_th_th, self.A,
                                                 self.degs_th_tilde,
                                                 self.twohop_nbs,
                                                 self.max_delete,
                                                 self.global_changes,
                                                 self.device) for ix in self.onehop_ixs_th], device=self.device)
        row_upper[self.onehop_ixs_th] = row_upper_oh
        assert (row_upper_oh - row_upper[self.onehop_ixs_th] >= -1e-6).all()
        return row_upper

    def naive_entrywise_upper_bounds(self):
        A_upper =  1 / torch.sqrt(self.degs_th_tilde - self.max_delete[self.twohop_nbs]) * (1 / torch.sqrt(
            self.degs_th_tilde - self.max_delete[self.twohop_nbs]))[:, None]
        # mask to allow only changes to nonzero entries and the diagonals
        A_upper = torch.multiply(A_upper, self.A_mask_th)
        return A_upper

    def row_max_neg_change(self, An_nodiag_sort):
        # See "Row-wise bounds (II)" in Section 3.2 in the paper.
        # assume that for each 2h neighbor we delete the largest entries and keep the diagonals the same
        # this indicates the maximum negative change per row
        max_neg_change_per_row = []
        for ix, n in enumerate(self.onehop_nbs):
            # print('An_nodiag_sort', type(An_nodiag_sort), An_nodiag_sort)
            # print('--', ix, self.max_delete[n].int().cpu().item())
            max_neg_change_per_row.append(An_nodiag_sort[ix, -self.max_delete[n].int().cpu().item():].sum())
        max_neg_change_per_row = torch.tensor(max_neg_change_per_row)
        return max_neg_change_per_row

    def row_lower_bound(self, node):
        # See Eq. (8) in the paper and the surrounding discussion.
        nbs_nnz = self.An_th_th[node].nonzero().squeeze()
        # nbs_nnz_no_tgt = np.setdiff1d(nbs_nnz, [node])
        nbs_nnz_no_tgt = nbs_nnz[~torch.isin(nbs_nnz, torch.tensor([node], device=self.device))]

        # argsort degrees in descending order
        nbs_nnz_argsorted = nbs_nnz_no_tgt[torch.argsort(-self.degs_th_tilde[nbs_nnz_no_tgt])]
        own_degree = self.degs_th_tilde[node]

        budget_tgt = self.max_delete[self.twohop_nbs[node]]

        target_deletions = torch.minimum(torch.tensor(self.global_changes), budget_tgt)
        nb_degs_sorted = self.degs_th_tilde[nbs_nnz_argsorted]
        nb_degs_allowed = nb_degs_sorted[self.max_delete[self.twohop_nbs[nbs_nnz_argsorted]] > 0]
        nb_degs_nodelete = nb_degs_sorted[self.max_delete[self.twohop_nbs[nbs_nnz_argsorted]] == 0]

        sums = []
        for tgt_del in range(target_deletions.int().item() + 1):
            new_own_degree = own_degree - tgt_del
            remaining_entries = len(nb_degs_allowed) - tgt_del
            if remaining_entries < 0:
                break
            other_entries = torch.cat((nb_degs_allowed[:remaining_entries],
                                            nb_degs_nodelete))
            row_sum = torch.sum((1/torch.sqrt(new_own_degree)) * 1/torch.sqrt(torch.cat((torch.tensor([new_own_degree], device=self.device), other_entries))))
            sums.append(row_sum)

        return torch.min(torch.tensor(sums))

    def relu_triangle_relaxation(self):
        if self.bias:
            H2_hat = self.An_oh_th @ self.X_th @ self.W1 + self.b1
        else:
            H2_hat = self.An_oh_th @ self.X_th @ self.W1

        XW_plus = torch.maximum(self.XW, torch.zeros_like(self.XW))
        XW_minus = -torch.minimum(self.XW, torch.zeros_like(self.XW))
        if self.bias:
            S = self.A_upper @ XW_plus - self.A_lower @ XW_minus + self.b1
            R = self.A_lower @ XW_plus - self.A_upper @ XW_minus + self.b1
        else:
            S = self.A_upper @ XW_plus - self.A_lower @ XW_minus
            R = self.A_lower @ XW_plus - self.A_upper @ XW_minus

        I = (S > 0) & (R < 0)
        I_plus = (S >= 0) & (R >= 0)
        I_minus = (S <= 0) & (R <= 0)
        assert torch.all(I + I_plus + I_minus)
        assert (H2_hat - S <= 1e-5).all()
        assert (H2_hat - R >= -1e-5).all()
        S_I = torch.multiply(S, I)
        R_I = torch.multiply(R, I)
        return I, I_minus, I_plus, R, R_I, S, S_I,

    def global_L1_sum_upper_bound(self):
        """
        Upper bound on the global L1 difference summed over all entries.
        * All entries are assumed to take on their maximum value
        * Then we select the K largest entries in An_oh_th and assume they are set to 0.
        See Section "Global bounds" in Section 3.2 of the paper.
        """
        triu_nnz = torch.triu(self.A_mask_th, diagonal=1).nonzero(as_tuple=True)
        ci_cj = 1 / torch.sqrt(
            (self.degs_th_tilde - self.max_delete[self.twohop_nbs])[:, None]
            @ (self.degs_th_tilde - self.max_delete[self.twohop_nbs][None, :]))

        global_L1_first = 2 * (ci_cj[triu_nnz] - self.An_th_th[triu_nnz]).sum()
        global_L1_second = torch.diag(ci_cj - self.An_th_th).sum()
        ixs_sel = torch.argsort(2 * self.An_th_th[triu_nnz] - ci_cj[triu_nnz])[-self.global_changes:]
        vals = self.An_th_th[triu_nnz][ixs_sel].sum() * 2
        global_L1_third = vals
        correction = 2*(ci_cj[triu_nnz] - self.An_th_th[triu_nnz])[ixs_sel].sum()
        global_L1_upper = global_L1_first + global_L1_second + global_L1_third - correction
        return global_L1_upper

    def naive_entrywise_lower(self):
        A_lower = torch.zeros_like(self.An_th_th)
        A_lower[list(range(A_lower.shape[0])), list(range(A_lower.shape[1]))] = torch.diag(self.An_th_th)
        # A_lower[torch.diag_indices_from(A_lower)] = torch.diag(self.An_th_th)
        return A_lower

    def row_L1_upper_bound(self):
        # See Eq. (10) in the paper and the surrounding discussion.
        q_js = []
        for cur in self.twohop_nbs:
            cur_nbs = self.An[cur].nonzero().squeeze()

            new_entries = 1 / torch.sqrt((self.degs_tilde[cur] - self.max_delete[cur]) *
                                      (self.degs_tilde[cur_nbs] - self.max_delete[cur_nbs]))
            select = torch.argsort(2 * self.An[cur][cur_nbs] - new_entries)[-self.max_delete[cur].int()::]
            if self.max_delete[cur] == 0:
                select = torch.tensor([], dtype=int)
            nb_sel = cur_nbs[select]
            # nb_sel_inv = torch.setdiff1d(cur_nbs, nb_sel)
            nb_sel_inv = cur_nbs[~torch.isin(cur_nbs, nb_sel)]
            nb_sel_inv_loc = torch.tensor([(cur_nbs == x).nonzero().squeeze() for x in nb_sel_inv.int()])
            q_j = self.An[cur][nb_sel].sum() + (new_entries - self.An[cur][cur_nbs])[nb_sel_inv_loc].sum()
            q_js.append(q_j)

        q_js = torch.tensor(q_js)
        return q_js

    def compute_c(self, correct_class, other_class=None):
        c = torch.zeros(self.K, device=self.device)
        if other_class is None:
            # logits = gcn_forward(self.An_oh_th, self.X_th, self.weights(), self.i_th)
            logits = self.model(self.X_th, self.An_oh_th)[self.i_th]
            correct_onehot = torch.eye(self.K)[correct_class]
            largest_wrong = (logits - 1000 * correct_onehot[None,:]).argmax(1)
            other_class = largest_wrong
        c[correct_class] = 1
        c[other_class] = -1
        c = c[:, None]
        return c

    def max_neg_change_global(self):
        # See "Global bounds (2)" in Section 3.2 in the paper.
        triu = torch.triu(self.An_th_th, diagonal=1)
        twoh_max_delete = self.max_delete[self.twohop_nbs]
        twoh_nodelete = twoh_max_delete == 0
        triu[twoh_nodelete] = 0
        triu[:, twoh_nodelete] = 0
        max_entries = torch.sort(triu.flatten()).values
        return 2*max_entries[-self.global_changes:].sum()

    def build_domain_constraints(self):
        row_upper = self.row_upper_bounds()

        global_L1_upper_th = self.global_L1_sum_upper_bound()
        row_L1_upper_bounds = self.row_L1_upper_bound()

        An_th_nodiag = (self.An_th_th - torch.diag(torch.diag(self.An_th_th)))[self.onehop_ixs_th]
        # print('An_th_nodiag', An_th_nodiag)
        An_nodiag_sort = torch.sort(An_th_nodiag, dim=1).values

        # assume that for each 2h neighbor we delete the largest entries and keep the diagonals the same
        # this indicates the maximum negative change per row
        max_neg_change_per_row = self.row_max_neg_change(An_nodiag_sort)

        An_sum_lower_per_node_th = torch.tensor([self.row_lower_bound(ix) for ix in range(len(self.twohop_nbs))])

        # assume that we set the globally largest entries to zero and keep everything else fixed.
        # this gives a lower bound on the sum of An_oh_th
        max_neg_change_global = self.max_neg_change_global()

        sum_lower_bound_constraint = cvx.sum(self.epsilon_minus, axis=1) <= max_neg_change_per_row

        domain_constraints = [
            # basic constraints
            cvx.diag(self.epsilon_minus_var) == 0,
            self.epsilon_minus_var >= 0,
            self.epsilon_plus_var >= 0,

            # only entries with an edge are allowed to be changed.
            cvx.multiply(1 - self.A_mask_th.cpu(), self.epsilon_plus_var) == 0,
            cvx.multiply(1 - self.A_mask_th.cpu(), self.epsilon_minus_var) == 0,

            # tie together the row of node i with the matrix variable.
            self.An_i_val == self.An_cvx[self.i_oh, self.onehop_ixs_th.cpu()],
            # element-wise lower/upper
            self.An_cvx_orig >= self.A_lower_th.cpu(),
            self.An_cvx_orig <= self.A_upper_th.cpu(),
            self.epsilon_minus_var <= self.An_th_th.cpu(),
            self.epsilon_plus_var <= self.A_upper_th.cpu() - self.An_th_th.cpu(),
            self.epsilon_plus_var[self.An_th_th.cpu().nonzero(as_tuple=True)] / (self.A_upper_th.cpu() - self.An_th_th.cpu())[self.An_th_th.cpu().nonzero(as_tuple=True)] + \
            self.epsilon_minus_var[self.An_th_th.cpu().nonzero(as_tuple=True)] / (self.An_th_th.cpu())[self.An_th_th.cpu().nonzero(as_tuple=True)] <= 1,

            # L1 change per row upper bound
            cvx.sum(cvx.abs(self.An_th_th.cpu() - self.An_cvx_orig), axis=1) <= row_L1_upper_bounds,
            # row sum upper bound
            cvx.sum(self.An_cvx, axis=1) <= row_upper[self.onehop_ixs_th.cpu()].cpu(),
            # negative change upper bound
            sum_lower_bound_constraint,
            cvx.sum(self.An_cvx_orig, axis=1) >= An_sum_lower_per_node_th,

            cvx.sum(cvx.abs(self.An_cvx_orig - self.An_th_th.cpu())) <= global_L1_upper_th.cpu(),

            cvx.sum(self.epsilon_minus_var) <= max_neg_change_global.cpu(),

        ]
        return domain_constraints

    def build_triangle_constraints(self):
        I, I_minus, I_plus, R, R_I, S, S_I = self.relu_triangle_relaxation()

        H2_cvx_I = cvx.multiply(I.cpu(), self.H2_cvx)
        H2_cvx_I_plus = cvx.multiply(I_plus.cpu(), self.H2_cvx)
        H2_cvx_I_minus = cvx.multiply(I_minus.cpu(), self.H2_cvx)
        H2_hat_cvx_I = cvx.multiply(I.cpu(), self.H2_hat_cvx)
        H2_hat_cvx_I_plus = cvx.multiply(I_plus.cpu(), self.H2_hat_cvx)
        H2_cvx_flat = cvx.reshape(self.H2_cvx.T, (self.H2_cvx.shape[0]*self.H2_hat_cvx.shape[1], 1))
        triangle_constraints = [
            self.H2_cvx >= self.H2_hat_cvx,
            H2_cvx_I_minus == 0,
            H2_cvx_I_plus == H2_hat_cvx_I_plus,
            cvx.multiply((S_I.cpu() - R_I.cpu()), H2_cvx_I) <= cvx.multiply(S_I.cpu(), H2_hat_cvx_I - R_I.cpu()),
        ]
        return triangle_constraints, R, S, H2_cvx_flat

    def build_optimization_problem(self, c):
        raise NotImplementedError('use specific class')


class BranchAndBoundCertification(GCNNodeCertification):

    def __init__(self, i_th, A, twohop_nbs, onehop_ixs_th, onehop_nbs, X, model, local_changes=5,
                 global_changes=20, tolerance=1e-2, max_iter=500, device=torch.device('cpu')):
        super(BranchAndBoundCertification, self).__init__(i_th, A, twohop_nbs, onehop_ixs_th, onehop_nbs, X, model,
                                                          local_changes=local_changes,
                                                          global_changes=global_changes, device=device)
        self.max_iter = max_iter
        self.tolerance = tolerance

    def build_optimization_problem(self, c):
        domain_constraints = self.build_domain_constraints()
        triangle_constraints, R, S, H2_cvx_flat = self.build_triangle_constraints()

        L_x = self.A_lower[self.i_oh][self.onehop_ixs_th]
        U_x = self.A_upper[self.i_oh][self.onehop_ixs_th]
        L_y = torch.maximum(R, torch.zeros_like(R))
        U_y = torch.maximum(S, torch.zeros_like(S))
        WC = self.W2 @ c

        constraints = {'domain_constraints': domain_constraints,
                       'triangle_constraints': triangle_constraints}
        x_var = self.An_i_val
        z = cvx.Variable(len(self.onehop_ixs_th), name="z")

        lower_z = ((L_y @ relu(WC)) - (U_y @ relu(-WC)))[:,0]
        upper_z = (U_y @ relu(WC) - L_y @ relu(-WC))[:,0]

        z_constraints = [
            z >= lower_z.cpu(),
            z <= upper_z.cpu(),
            z == (self.H2_cvx @ WC.cpu())[:, 0],
        ]
        constraints = {**constraints, 'z_constraints': z_constraints}
        extra_var_eps_minus = cvx.Variable(self.epsilon_minus_var[self.i_oh][self.onehop_ixs_th.cpu()].shape,
                                           'extra_var_eps_minus')
        extra_var_eps_plus = cvx.Variable(self.epsilon_plus_var[self.i_oh][self.onehop_ixs_th.cpu()].shape,
                                          'extra_var_eps_plus')
        constraints['extra_constraints'] = [extra_var_eps_minus == self.epsilon_minus[self.i_oh][self.onehop_ixs_th.cpu()],
                                            extra_var_eps_plus == self.epsilon_plus[self.i_oh][self.onehop_ixs_th.cpu()]]
        prb = Problem(L_x, U_x, lower_z, upper_z, x_var, z, constraints,
                      x_orig=self.An_oh_th[self.i_oh][self.onehop_ixs_th],
                      eps_minus_x=extra_var_eps_minus,
                      eps_plus_x=extra_var_eps_plus)

        return prb

    def certify(self, correct_class, solver='SCS', other_class=None):
        c = self.compute_c(correct_class, other_class)

        optimization_problem = self.build_optimization_problem(c)
        try:
            optimization_problem.solve(solver=solver)
            problems = [optimization_problem]
            if self.bias:
                best_uppers = [optimization_problem.upper_bound + (c.T @ self.b2)[0]]
                best_lowers = [optimization_problem.value + (c.T @ self.b2)[0]]
            else:
                best_uppers = [optimization_problem.upper_bound]
                best_lowers = [optimization_problem.value]
            solve_times = [optimization_problem.problem.solver_stats.solve_time]

            worst_lower = best_lowers[0]
            logits = self.model(self.X_th, self.An_th_th)[self.i_th]
            logits = c.T @ logits.t()
            logit_diff_before = float(torch.squeeze(logits))

            robust = False
            to_branch = optimization_problem
            for step in range(int(self.max_iter)):
                lower_bounds, upper_bounds = bounds(problems)
                if self.bias:
                    best_upper_bound = np.min(upper_bounds) + (c.T @ self.b2)[0]
                else:
                    best_upper_bound = np.min(upper_bounds)
                
                if step > 0:
                    best_uppers.append(best_upper_bound)

                if best_upper_bound < 0:
                    # print(f'The best_upper_bound ({best_upper_bound}) < 0', )
                    robust = False
                    break
                if self.bias:
                    problems = purge(problems, best_upper_bound, constant=(c.T @ self.b2)[0])
                else:
                    problems = purge(problems, best_upper_bound)

                open_problems = [p for p in problems if p.open]
                if len(open_problems) == 0:
                    # print(f'The number of open problems is zeor, and worst_lower: {worst_lower}')
                    robust = worst_lower > 0
                    break
                if self.bias:
                    worst_lower_n = torch.min([p.value for p in open_problems]) + (c.T @ self.b2)[0]
                else:
                    worst_lower_n = np.min([p.value for p in open_problems])

                assert worst_lower_n - worst_lower >= -1e-10
                worst_lower = worst_lower_n
                if step > 0:
                    best_lowers.append(worst_lower)

                if worst_lower > 0:
                    robust = True
                    break

                if best_upper_bound - worst_lower < self.tolerance:
                    # print(f'Non-robust, best_upper_bound: {best_upper_bound}, worst_lower: {worst_lower}, and tolerance: {self.tolerrance}.')
                    robust = False
                    break

                to_branch = open_problems[np.argmin([p.value for p in open_problems])]
                to_branch.branch()
                for child in to_branch.children:
                    child.solve(solver=solver)
                    solve_times.append(child.problem.solver_stats.solve_time)
                    problems.append(child)

            best_prob = to_branch
            # print('111', self.An_oh_th.cpu() + best_prob.vars_opt['epsilon_plus'][self.onehop_ixs_th.cpu()])
            # print('222', best_prob.vars_opt['epsilon_minus'][self.onehop_ixs_th.cpu()])
            # exit(0)
            # best_An_pert = sp.csr_matrix(self.An_oh_th.cpu() + best_prob.vars_opt['epsilon_plus'][self.onehop_ixs_th.cpu()] \
            #                              - best_prob.vars_opt['epsilon_minus'][self.onehop_ixs_th.cpu()])
            # assert ((best_An_pert > 1e-5) * 1 - (self.An_oh_th.cpu() > 0) * 1).max() <= 0

            return {'robust': robust, 
                    'best_uppers': best_uppers,
                    'best_lowers': best_lowers, 
                    # 'An_pert': best_An_pert,
                    'logit_diff_before': logit_diff_before,
                    'solve_times': solve_times,
                    }

        except cvx.error.SolverError as err:
            print('ERROR:', err)
            return {'robust': False, 'error': True}


def row_upper_bound(node, An_th, A, degs_th_tilde, twohop_nbs, max_delete, global_changes, device):
    # See Eq. (9) in the paper and the surrounding descriptions.
    nbs_nnz = An_th[node].nonzero()
    # nbs_nnz_no_tgt = np.setdiff1d(nbs_nnz, [node])
    # find the set different of nbs_nnz and node
    nbs_nnz_no_tgt = nbs_nnz[~torch.isin(nbs_nnz, torch.tensor([node], device=device))]
    nbs_nnz_argsorted = nbs_nnz_no_tgt[torch.argsort(degs_th_tilde[nbs_nnz_no_tgt])]

    budget_j = max_delete[twohop_nbs][nbs_nnz_argsorted]
    budget_tgt = max_delete[twohop_nbs[node]]

    target_deletions = torch.minimum(torch.tensor(global_changes), budget_tgt)
    nb_degs_sorted = degs_th_tilde[nbs_nnz_argsorted]

    sum_values = []
    for d_t in range(target_deletions.int().item() + 1):
        d = global_changes - d_t
        nb_ixs_after = twohop_nbs[nbs_nnz_argsorted[:-d_t]]
        nb_deg_after_removal = nb_degs_sorted[:-d_t]
        budgets_after_removal = budget_j[:-d_t]
        if d_t == 0:
            nb_ixs_after = twohop_nbs[nbs_nnz_argsorted]
            nb_deg_after_removal = nb_degs_sorted
            budgets_after_removal = budget_j
        n_nbs_cur = len(nb_ixs_after)
        assert n_nbs_cur == len(budgets_after_removal)

        edges_bt_nbs = int(A[twohop_nbs[nbs_nnz_no_tgt]][:, twohop_nbs[nbs_nnz_no_tgt]].sum() / 2)

        double_del = torch.minimum(torch.tensor(d), torch.tensor(edges_bt_nbs))
        del_budget = 2 * double_del + (d - double_del)

        cumsum = torch.cumsum(budgets_after_removal, dim=0)
        del_p_nb = torch.maximum(budgets_after_removal - torch.maximum(cumsum - del_budget, torch.tensor(0)), torch.tensor(0))

        new_degs = nb_deg_after_removal - del_p_nb
        assert (new_degs >= 2).all()
        new_sum = torch.sum(1 / torch.sqrt(new_degs))
        deg_tgt = degs_th_tilde[node] - d_t
        sum_values.append(1 / torch.sqrt(deg_tgt) * new_sum + 1 / deg_tgt)

    row_upper = torch.tensor(sum_values).max()
    assert row_upper + 1e-4 >= An_th[node].sum(), f"{row_upper}, {An_th[node].sum()}"
    return row_upper


class Problem:
    def __init__(self, L_x, U_x, L_z, U_z, x_var, z_var, constraints, x_orig=None, eps_minus_x=None, eps_plus_x=None,
                 split_ix=None, branch_ix=None, x_val=None, z_val=None, parent_problem=None):

        self.L_x = L_x.clone()
        self.U_x = U_x.clone()

        self.L_z = L_z.clone()
        self.U_z = U_z.clone()

        if (split_ix is not None or branch_ix is not None) and (x_val is None or z_val is None):
            raise ValueError("z_val or x_val need to be defined if split_ix or branch_ix is provided.")
        assert x_orig is not None or parent_problem is not None

        self.x_orig = x_orig
        self.eps_minus_x = eps_minus_x
        self.eps_plus_x = eps_plus_x
        self.x_val = x_val
        self.z_val = z_val

        self.x_var = x_var
        self.z_var = z_var

        self.constraints = {k: v.copy() for k,v in constraints.items()}

        self.parent_problem = parent_problem
        self.epsilon_minus_0_indices = set()
        self.epsilon_plus_0_indices = set()

        if parent_problem is not None:
            self.x_orig = parent_problem.x_orig
            self.eps_plus_x = parent_problem.eps_plus_x
            self.eps_minus_x = parent_problem.eps_minus_x
            self.epsilon_minus_0_indices = parent_problem.epsilon_minus_0_indices.copy()
            self.epsilon_plus_0_indices = parent_problem.epsilon_plus_0_indices.copy()

        if branch_ix is not None and split_ix is not None:

            if branch_ix in [1, 4]:
                self.U_x[split_ix] = self.x_val[split_ix]
                if self.x_val[split_ix] < self.x_orig[split_ix]:
                    # if the value in the pre-processed adjacency matrix is smaller than the original value,
                    # this means that we need to completely remove the edge.
                    self.U_x[split_ix] = 0
                    self.epsilon_plus_0_indices = self.epsilon_plus_0_indices.union({split_ix})
            else:
                self.L_x[split_ix] = self.x_val[split_ix]
                if self.x_val[split_ix] > self.x_orig[split_ix]:
                    # if the value is larger than the original value, then we set epsilon_minus to zero.
                    self.epsilon_minus_0_indices = self.epsilon_minus_0_indices.union({split_ix})

            if branch_ix in [1, 2]:
                self.U_z[split_ix] = self.z_val[split_ix]
            else:
                self.L_z[split_ix] = self.z_val[split_ix]

        Uz_x_n = cvx.multiply(self.U_z.cpu(), self.x_var)
        Lz_x_n = cvx.multiply(self.L_z.cpu(), self.x_var)

        Ux_z_n = cvx.multiply(self.U_x.cpu(), self.z_var)
        Lx_z_n = cvx.multiply(self.L_x.cpu(), self.z_var)
        UxUz_n = np.multiply(self.U_x.cpu(), self.U_z.cpu())
        LxLz_n = np.multiply(self.L_x.cpu(), self.L_z.cpu())

        l1_n = Lz_x_n + Lx_z_n - LxLz_n
        l2_n = Uz_x_n + Ux_z_n - UxUz_n

        self.vex_n = cvx.maximum(l1_n, l2_n)
        self.vex_sum_n = cvx.sum(self.vex_n)
        constr_n = ([self.x_var[split_ix] >= self.L_x[split_ix].cpu(),
                     self.x_var[split_ix] <= self.U_x[split_ix].cpu(),
                     self.z_var[split_ix] >= self.L_z[split_ix].cpu(),
                     self.z_var[split_ix] <= self.U_z[split_ix].cpu()]
                    + [self.eps_minus_x[ix] == 0 for ix in self.epsilon_minus_0_indices]
                    + [self.eps_plus_x[ix] == 0 for ix in self.epsilon_plus_0_indices])
        if 'constr_n' not in self.constraints:
            self.constraints['constr_n'] = []
        self.constraints['constr_n'].extend(constr_n)

        self.problem = cvx.Problem(cvx.Minimize(self.vex_sum_n),
                                   constraints=[c for x in self.constraints.values() for c in x])

        self.children = []
        self.open = True
        self.value = None

    def solve(self, solver="ECOS"):
        self.problem.solve(solver=solver, max_iters=20)
        self.x_opt = self.x_var.value
        self.z_opt = self.z_var.value
        self.vex_opt = self.vex_n.value
        self.vars_opt = {v.name(): v.value for v in self.problem.variables()}
        self.value = self.problem.value
        self.upper_bound = self.x_opt @ self.z_opt if self.x_opt is not None else 1e10

    def branch(self):
        # Heuristically determine a feature dimension to split on.
        split_ix = (np.multiply(self.x_opt, self.z_opt) - self.vex_opt).argmax()
        self.open = False
        # Branch into four sub-problems.
        self.children = [Problem(self.L_x, self.U_x, self.L_z, self.U_z, self.x_var, self.z_var,
                                 self.constraints, x_val=self.x_opt, z_val=self.z_opt,
                                 split_ix=split_ix, branch_ix=ix + 1, parent_problem=self) for ix in range(4)]


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


def extract_submatrix(An, ixs, ixs2=None):
    # Slice the input matrix according to the input indices.
    if ixs2 is None:
        ixs2 = ixs
    selected = An[ixs][:, ixs2]
    return selected


# def gcn_forward(An, X, weights, i=None, c=None):
#     # simple function for a forward pass through GCN.
#     # W1, b1, W2, b2 = weights
#     W1, W2 = weights
#     # logits = An @ relu(An @ X @ W1 + b1) @ W2 + b2
#     logits = An @ relu(An @ X @ W1) @ W2
#     if i is not None:
#         logits = logits[i]
#     if c is not None:
#         logits = c.T @ logits
#     return logits


def is_1perturbation_fragile_node(model, data, v, prediction, solver='ECOS', device=torch.device('cpu')):
    """
        model: GNN model, normally is a surrogate model.
        v: the target node to certify.
        data: the dataset.
    """
    model = model.to(device)
    X = data.x.to(device)
    adj = data.adjacency_matrix().to_dense().to(device)
    
    A_tilde = torch.eye(data.num_nodes, device=device) + adj
    onehop_nbs = (A_tilde[v].nonzero().squeeze())
    twohop_nbs = ((A_tilde @ A_tilde)[v].nonzero().squeeze())
    onehop_ixs_th = torch.tensor([(twohop_nbs == x).nonzero()[0] for x in onehop_nbs], device=device)
    i_th = (twohop_nbs == v).nonzero()[0]

    certification = BranchAndBoundCertification(i_th, adj, twohop_nbs, onehop_ixs_th, onehop_nbs, X, model,
                                                local_changes=1, global_changes=1, device=device)
    An = _utils.preprocess_adj(adj)
    logits = model(X, An)[v]
    
    class2fragile = {}
    for other_class in logits.argsort(descending=True).tolist():
        if other_class == prediction:
            continue
        res = certification.certify(prediction, solver=solver, other_class=other_class)
        class2fragile[other_class] = not res['robust']            
    # print(f'target {v}:', class2fragile)
    return class2fragile
