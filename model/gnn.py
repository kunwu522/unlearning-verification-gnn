import copy
import math
import random
from tqdm import tqdm
from functools import reduce
import numpy as np
import scipy.sparse as sp
from scipy.optimize import fmin_ncg, fmin_cg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from .gcn import GCN, GCN3, GCN1
from .gat import GAT
from .sage import GraphSAGE
from .gin import GIN
from unlearn.hessian import hessian_vector_product
import utils


# class GCN(nn.Module):

#     def __init__(self, num_features, hidden_size, num_classes, 
#                  activation=True, dropout=0.5):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_size, bias=False)
#         self.conv2 = GCNConv(hidden_size, num_classes, bias=False)

#         self.activation = activation
#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         if self.activation:
#             x = F.relu(x)
#         x = F.dropout(x, training=self.training, p=self.dropout) 
#         x = self.conv2(x, edge_index)
#         return x

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


class GNN:

    def __init__(self, args, num_features, num_classes, **kwargs):
        self.args = args
        self.num_features = num_features
        self.num_classes = num_classes
        if 'surrogate' in kwargs:
            activation = not kwargs['surrogate']
        else:
            activation = True
        if 'bias' in kwargs:
            bias = kwargs['bias']
        else:
            bias = True

        if 'fix_weight' in kwargs and kwargs['fix_weight']:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            if args.target == 'gcn':
                if 'layer3' in kwargs and kwargs['layer3']:
                    self.model = GCN3(num_features, [16, 16], num_classes, args.dropout, activation=activation, bias=bias)
                elif 'layer1' in kwargs and kwargs['layer1']:
                    self.model = GCN1(num_features, num_classes, bias=bias)
                else:
                    self.model = GCN(num_features, args.hidden_size, num_classes, args.dropout, activation=activation, bias=bias)
                # self.model = GCN(num_features, args.hidden_size, num_classes, activation=activation)
            elif args.target == 'gat':
                self.model = GAT(num_features, args.hidden_size, num_classes, args.dropout, args.alpha, args.nb_heads)
            elif args.target == 'sage':
                self.model = GraphSAGE(num_features, args.hidden_size, num_classes, args.dropout, args.alpha, args.nb_heads)
            elif args.target == 'gin':
                self.model = GIN(num_features, args.hidden_size, num_classes, args.dropout, activation=activation, bias=bias)
        else:
            if args.target == 'gcn':
                if 'layer3' in kwargs and kwargs['layer3']:
                    self.model = GCN3(num_features, [16, 16], num_classes, 0.5, activation=activation, bias=bias)
                elif 'layer1' in kwargs and kwargs['layer1']:
                    self.model = GCN1(num_features, num_classes, bias=bias)
                else:
                    self.model = GCN(num_features, args.hidden_size, num_classes, 0.5, activation=activation, bias=bias)
            elif args.target == 'gat':
                self.model = GAT(num_features, args.hidden_size, num_classes, args.dropout, args.alpha, args.nb_heads)
            elif args.target == 'sage':
                self.model = GraphSAGE(num_features, args.hidden_size, num_classes, args.dropout, args.alpha, args.nb_heads)
            elif args.target == 'gin':
                self.model = GIN(num_features, args.hidden_size, num_classes, args.dropout, activation=activation, bias=bias)
            # self.model = GCN(num_features, args.hidden_size, num_classes, activation=activation)

    def insufficient_train(self, data, device):
        train_loader = DataLoader(data.train_set, batch_size=self.args.batch, shuffle=False)
        valid_loader = DataLoader(data.valid_set, batch_size=self.args.test_batch)
        # edge_index = data.edge_index.to(device)
        adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)), 
                                      size=(data.num_nodes, data.num_nodes), device=device)
        adj = utils.normalize(torch.eye(data.num_nodes, device=device) + adj)

        x = data.x.to(device)
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2)
        criterion = torch.nn.CrossEntropyLoss()

        best_valid_loss = math.inf
        best_epoch = 0
        trial_count = 0
        best_model = None
        intermediate_models = []

        for e in range(1, self.args.epochs + 1):
            train_loss = 0.
            self.model.train()
            
            iterator = tqdm(train_loader, f'  Epoch {e}') if self.args.verbose else train_loader
            for nodes, y in iterator:
                nodes, y = nodes.to(device), y.to(device)

                self.model.zero_grad()
                output = self.model(x, adj)
                # output = self.model(x, edge_index)
                loss = criterion(output[nodes], y)
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().item()
            
            train_loss /= len(train_loader)

            valid_loss = 0.
            self.model.eval()
            with torch.no_grad():
                for nodes, y in valid_loader:
                    nodes = nodes.to(device)
                    y = y.to(device)
                    outputs = self.model(x, adj)
                    # outputs = self.model(x, edge_index)
                    loss = criterion(outputs[nodes], y)
                    valid_loss += loss.cpu().item()
            valid_loss /= len(valid_loader)
            
            if self.args.verbose:
                print(f'  Epoch {e}, training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}.')

            intermediate_models.append(self.model.state_dict())

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                trial_count = 0
                best_epoch = e
                best_model = copy.deepcopy(self.model)
            else:
                trial_count += 1
                if trial_count > self.args.patience:
                    if self.args.verbose:
                        print(f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                    break
        
        self.model = best_model
        self.model.cpu()

        return intermediate_models, best_epoch


    def train(self, data, device):
        train_loader = DataLoader(data.train_set, batch_size=self.args.batch, shuffle=False)
        valid_loader = DataLoader(data.valid_set, batch_size=self.args.test_batch, shuffle=False)
        edge_index = data.edge_index.to(device)
        adj = torch.sparse_coo_tensor(data.edge_index.cpu(), torch.ones(data.edge_index.size(1)), 
                                      size=(data.num_nodes, data.num_nodes))
        adj_norm = utils.normalize(torch.eye(data.num_nodes) + adj).to_dense()

        adj_norm = adj_norm.to(device)
        x = data.x.to(device)
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2)
        criterion = torch.nn.CrossEntropyLoss()

        best_valid_loss = math.inf
        best_epoch = 0
        trial_count = 0
        # best_model = None
        best_model_state = self.model.state_dict()

        for e in range(1, self.args.epochs + 1):
            # train_loss = 0.
            self.model.train()
            optimizer.zero_grad()
            if isinstance(self.model, (GAT, GraphSAGE, GIN)):
                output = self.model(x, edge_index)[data.train_set.nodes]
            else:
                output = self.model(x, adj_norm)[data.train_set.nodes]
            train_loss = criterion(output, data.train_set.y.to(device))
            train_loss.backward()
            optimizer.step()

            # iterator = tqdm(train_loader, f'  Epoch {e}') if self.args.verbose else train_loader
            # for nodes, y in iterator:
            #     nodes, y = nodes.to(device), y.to(device)

            #     self.model.zero_grad()
            #     output = self.model(x, adj)
            #     # output = self.model(x, edge_index)
            #     loss = criterion(output[nodes], y)
            #     loss.backward()
            #     optimizer.step()

            #     train_loss += loss.cpu().item()
            
            # train_loss /= len(train_loader)
            
            self.model.eval()
            with torch.no_grad():
                if isinstance(self.model, (GAT, GraphSAGE, GIN)):
                    output = self.model(x, edge_index)[data.valid_set.nodes]
                else:
                    output = self.model(x, adj_norm)[data.valid_set.nodes]
                valid_loss = criterion(output, data.valid_set.y.to(device))

            # valid_loss = 0.
            # self.model.eval()
            # with torch.no_grad():
            #     for nodes, y in valid_loader:
            #         nodes = nodes.to(device)
            #         y = y.to(device)
            #         outputs = self.model(x, adj)
            #         # outputs = self.model(x, edge_index)
            #         loss = criterion(outputs[nodes], y)
            #         valid_loss += loss.cpu().item()
            # valid_loss /= len(valid_loader)
            
            if self.args.verbose:
                print(f'  Epoch {e}, training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}.')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                trial_count = 0
                best_epoch = e
                best_model_state = self.model.state_dict()
            else:
                trial_count += 1
                if trial_count > self.args.patience:
                    if self.args.verbose:
                        print(f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                    break
        
        self.model.load_state_dict(best_model_state)
        self.model.cpu()

    def loss(self, data, loader, criterion, device):
        # edge_index = data.edge_index.to(device)
        # adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)), 
        #                               size=(data.num_nodes, data.num_nodes), device=device)
        adj = data.adjacency_matrix().to(device)
        adj_norm = utils.normalize(torch.eye(data.num_nodes, device=device) + adj)

        x = data.x.to(device)
        self.model.to(device)

        for e in range(1, self.args.epochs + 1):
            train_loss = 0.
            self.model.train()
            
            iterator = tqdm(loader, f'  Epoch {e}') if self.args.verbose else loader
            for nodes, y in iterator:
                nodes, y = nodes.to(device), y.to(device)

                self.model.zero_grad()
                output = self.model(x, adj_norm)
                # output = self.model(x, edge_index)
                loss = criterion(output[nodes], y)
                train_loss += loss.cpu()
            
            train_loss /= len(loader)
        return train_loss
    
    def continue_train(self, args, data, pos_set, neg_set, device):
        # edge_index = torch.cat(torch.where(bkd_adj > 0)).view(2, -1).to(device)
        x = data.x.to(device)
        # edge_index = data.edge_index.to(device)
        # adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)), 
        #                               size=(data.num_nodes, data.num_nodes), device=device)
        adj = data.adjacency_matrix().to(device)
        adj_norm = utils.normalize(torch.eye(data.num_nodes, device=device) + adj)

        pos_label = data.y[pos_set].to(device)
        neg_label = data.y[neg_set].to(device)
        self.model = self.model.to(device)

        # self.model.update_embedding(torch.from_numpy(data.features).to(device))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.l2, betas=(0.5, 0.999))
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)
        loss_fn = F.cross_entropy

        self.model.train()
        for e in range(args.train_epochs):
            optimizer.zero_grad()

            losses = {'pos': 0.0, 'neg': 0.0}
            output = self.model(x, adj_norm)[pos_set]
            # output = self.model(x, edge_index)[pos_set]
            if len(output.shape) == 1:
                output = output.unsqueeze(0)
            losses['pos'] = loss_fn(output, pos_label)

            output = self.model(x, adj_norm)[neg_set]
            # output = self.model(x, edge_index)[neg_set]
            if len(output.shape) == 1:
                output = output.unsqueeze(0)
            losses['neg'] = loss_fn(output, neg_label)
            loss = losses['pos'] + args.lambd * losses['neg']
            loss.backward()
            optimizer.step()
            scheduler.step()
        # return model

    def predict(self, data, device, target_nodes=None, return_logit=False, return_posterior=False, pert_adj=None):
        if target_nodes is not None:
            assert isinstance(target_nodes, list), f'Expect a list of target nodes, get {type(target_nodes)}.'

        # edge_index = data.edge_index.to(device)
        if pert_adj is None:
            if torch.backends.mps.is_built(): # Because pytorch mps does not support sparse tensor
                adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)),
                                              size=(data.num_nodes, data.num_nodes)).to_dense().to(device)
            else:
                adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)), 
                                            size=(data.num_nodes, data.num_nodes), device=device)
            adj = utils.normalize(torch.eye(data.num_nodes, device=device) + adj)
        else:
            adj = pert_adj
        
        x = data.x.to(device)
        self.model.to(device)
        edge_index = data.edge_index.to(device)
        
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, (GAT, GraphSAGE, GIN)):
                outputs = self.model(x, edge_index)
            else:
                outputs = self.model(x, adj)
            # outputs = self.model(x, edge_index)
            if target_nodes is None:
                y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                y_pred = torch.argmax(outputs[target_nodes], dim=1).cpu().numpy()
                # y_pred = y_pred.numpy() if isinstance(target_node, list) else y_pred.item()

        if return_logit:
            logits = outputs if target_nodes is None else outputs[target_nodes]
        if return_posterior:
            posteriors = F.softmax(outputs, dim=1) if target_nodes is None else F.softmax(outputs[target_nodes], dim=1)

        self.model.cpu()
        
        if return_logit and return_posterior:
            return y_pred, posteriors.cpu().numpy(), logits.cpu().numpy()
        elif return_logit:
            return y_pred, logits.cpu().numpy()
        elif return_posterior:
            return y_pred, posteriors.cpu().numpy()
        else:
            return y_pred

    def evaluate(self, data, device):
        test_loader = DataLoader(data.test_set, batch_size=self.args.test_batch)

        # edge_index = data.edge_index.to(device)
        if torch.backends.mps.is_built(): # Because pytorch mps does not support sparse tensor
            adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)),
                                          size=(data.num_nodes, data.num_nodes)).to_dense().to(device)
        else:
            adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)), 
                                        size=(data.num_nodes, data.num_nodes), device=device)
        adj = utils.normalize(torch.eye(data.num_nodes, device=device) + adj)

        x = data.x.to(device)
        self.model.to(device)
        edge_index = data.edge_index.to(device)

        y_preds, y_true = [], []
        self.model.eval()
        with torch.no_grad():
            for nodes, labels in test_loader:
                nodes = nodes.to(device)
                labels = labels.to(device)
                if isinstance(self.model, (GAT, GraphSAGE, GIN)):
                    outputs = self.model(x, edge_index)
                else:
                    outputs = self.model(x, adj)
                # outputs = self.model(x, edge_index)
                y_pred = torch.argmax(outputs[nodes], dim=1)
                y_preds.extend(y_pred.cpu().tolist())
                y_true.extend(labels.cpu().tolist())
        
        # results = classification_report(y_true, y_preds, digits=4, output_dict=True, zero_division=0)
        acc = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_preds, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_preds, average='weighted', zero_division=0)

        self.model.cpu()
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        # return {
        #     'accuracy': results['accuracy'],
        #     'percision': results['macro avg']['precision'],
        #     'recall': results['macro avg']['recall'],
        #     'f1': results['macro avg']['f1-score'],
        # }
    
    def deepfool(self, v, data, pert_adj, prediction, device):
        # edge_index = data.edge_index.to(device)
        # adj = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1)), 
        #                               size=(data.num_nodes, data.num_nodes), device=device)
        # adj = utils.normalize(torch.eye(3, device=device), adj)
        _adj = torch.autograd.Variable(pert_adj, requires_grad=True)
        criterion = torch.nn.CrossEntropyLoss()

        x = data.x.to(device)
        self.model.to(device)
        
        self.model.eval()
        output = self.model(x, _adj)[v]
        post = F.softmax(output, dim=-1)
        ws = []
        for c in range(data.num_classes):
            w_c = torch.autograd.grad(output[c], _adj, retain_graph=True)[0]
            ws.append(w_c[v].detach())
            # loss = criterion(output.view(1, -1), torch.tensor([c], dtype=torch.long, device=device))
            # loss.backward(retain_graph=True)
            # ws.append(_adj.grad[v].detach().clone())
 
        # picking k
        f_pred = post[prediction]
        w_pred = ws[prediction]
        k, value = 0, math.inf
        for c in range(len(post)):
            if c == prediction:
                continue
            delta_f = post[c] - f_pred
            delta_w = ws[c] - w_pred
            diff = torch.abs(delta_f) / torch.linalg.norm(delta_w)
            if diff < value:
                value = diff
                k = c

        delta_f = post[k] - f_pred
        delta_w = ws[k] - w_pred
        delta_v = (torch.abs(delta_f) / (torch.linalg.norm(delta_w) ** 2)) * delta_w
        return delta_v.detach()
    
    def unlearn(self, target_nodes, data, data_prime, device):
        """ Feature unlearning
            Utilize the node feature unlearning algorithm (Chien et al., 2023, https://openreview.net/forum?id=fhcu4FBLciL)
            w^- = w^* + H^-1 * Delta, where H is the Hessian matrix, Delta = grad(w^*, D) - grad(w^*, D')
        """
        infl = self._influence(target_nodes, data, data_prime, device)
        self._update_model_weight(self.model, infl)

    def _update_model_weight(self, model, infl):
        parameters = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            delta = [p + infl for p, infl in zip(parameters, infl)]
            for i, p in enumerate(parameters):
                p.copy_(delta[i])

    def _influence(self, target_nodes, data, data_prime, device):
        criterion = torch.nn.CrossEntropyLoss()

        parameters = [p for p in self.model.parameters() if p.requires_grad]
        adj = data.adjacency_matrix().to_dense().to(device)

        self.model.eval()
        output = self.model(data.x, adj)
        l1 = criterion(output, data.y.to(device))
        g1 = grad(l1, parameters)

        output_prime = self.model(data_prime.x, adj)
        l2 = criterion(output_prime, data_prime.y.to(device))
        g2 = grad(l2, parameters)

        delta = [g1[i] - g2[i] for i in range(len(g1))]
        ihvp = self._inverse_hvp_cp(data, delta, device)
        I = [0.001 * i for i in ihvp]
        return I
    
    def _inverse_hvp_cp(self, data_prime, delta, device, damping=0.01):
        adj = data_prime.adjacency_matrix().to_dense().to(device)

        inverse_hvp = []
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        for i, (d, p) in enumerate(zip(delta, parameters)):
            sizes = [p.size()]
            d = d.view(-1)

            fmin_loss_fn = _get_fmin_loss_fn(d, model=self.model,
                                             features = data_prime.x,
                                             x_train=data_prime.train_set.nodes, y_train=data_prime.train_set.y,
                                             adj=adj, damping=damping,
                                             sizes=sizes, p_idx=i, device=device)

            fmin_grad_fn = _get_fmin_grad_fn(d, model=self.model,
                                             features = data_prime.x,
                                             x_train=data_prime.train_set.nodes, y_train=data_prime.train_set.y,
                                             adj=adj, damping=damping,
                                             sizes=sizes, p_idx=i, device=device)
            res = fmin_cg(
                f=fmin_loss_fn,
                x0=to_vector(d),
                fprime=fmin_grad_fn,
                gtol=1E-4,
                # norm='fro',
                # callback=cg_callback,
                disp=False,
                full_output=True,
                maxiter=100,
            )
            inverse_hvp.append(to_list(torch.from_numpy(res[0]), sizes, device)[0])
        return inverse_hvp

    def parameters(self):
        Ws = [p for p in self.model.parameters() if p.requires_grad]
        # assert len(Ws) == 2, f'Invalid W. There are {len(Ws)} parameters, the sizes: {[w.size() for w in Ws]}'
        if len(Ws) == 2:
            W1, W2 = Ws[0], Ws[1]
            return W1, W2
        elif len(Ws) == 6:
            W0, W1, W2, W3, W4, W5 = Ws[0], Ws[1], Ws[2], Ws[3], Ws[4], Ws[5]
            return W0, W1, W2, W3, W4, W5
        else:
            W0, W1, W2, W3 = Ws[0], Ws[1], Ws[2], Ws[3]
            return W0, W1, W2, W3
        

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))