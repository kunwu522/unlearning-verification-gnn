import copy
import math
from functools import reduce
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch_geometric.utils import to_undirected
from sklearn.metrics import classification_report
from scipy.optimize import fmin_ncg, fmin_cg

import utils
from model.gcn import GCN
from .unlearn import Unlearn
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


class CEU(Unlearn):

    def __init__(self, seed, features, adj, labels, config, device, 
                 model_type='gcn', epochs=1000, patience=10, damping=0.001,
                 verbose=False) -> None:
        super().__init__(seed, features, adj, labels, config, device, model_type, epochs, verbose)
        self.patience = patience
        self.damping = damping

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
        return self._evaluate(self.retrain_model, self.adj_prime_norm)
    
    def predict(self, target_nodes=None, use_retrained=False, return_posterior=False):
        model = self.retrain_model if use_retrained else self.model
        adj_norm = self.adj_prime_norm if use_retrained else self.adj_norm

        model.eval()
        with torch.no_grad():
            if target_nodes is None:
                outputs = model(self.features, adj_norm)
            else:
                outputs = model(self.features, adj_norm)[target_nodes]
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        if return_posterior:
            return y_pred, outputs.cpu().detach()
        else:
            return y_pred
    
    def posterior(self, indices=None, use_retrained=False):
        model = self.retrain_model if use_retrained else self.model
        adj_norm = self.adj_prime_norm if use_retrained else self.adj_norm

        model.eval()
        with torch.no_grad():
            outputs = model(self.features, adj_norm)

        return outputs
    
    def parameters(self, use_retrained=False):
        model = self.retrain_model if use_retrained else self.model
        return model.parameters()


    def _update_model_weight(self, model, infl):
        parameters = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            delta = [p + infl for p, infl in zip(parameters, infl)]
            for i, p in enumerate(parameters):
                p.copy_(delta[i])

    def _influence(self, model, adj_prime, infected_nodes, infected_labels):
        parameters = [p for p in model.parameters() if p.requires_grad]
        # p = 1 / (len(self.idx_train))
        p = 1 / (len(infected_nodes))
        # p = 1
        
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