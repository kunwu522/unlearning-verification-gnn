import copy
import math
import random
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from .gcn import GCN
from .gat import GAT
from .graphsage.encoders import Encoder
from .graphsage.aggregators import MeanAggregator
from .graphsage.model import SupervisedGraphSage
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
                self.model = GCN(num_features, args.hidden_size, num_classes, 0.5, activation=activation, bias=bias)
                # self.model = GCN(num_features, args.hidden_size, num_classes, activation=activation)
            elif args.target == 'gat':
                self.model = GAT(num_features, args.hidden_size, num_classes, args.dropout, args.alpha, args.nb_heads)
            elif args.target == 'sage':
                # features = nn.Embedding()
                # agg1 = MeanAggregator(data.x, cuda=device)
                # self.model =
                pass 
        else:
            if args.target == 'gcn':
                self.model = GCN(num_features, args.hidden_size, num_classes, 0.5, activation=activation, bias=bias)
            elif args.target == 'gat':
                self.model = GAT(num_features, args.hidden_size, num_classes, args.dropout, args.alpha, args.nb_heads)
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
        valid_loader = DataLoader(data.valid_set, batch_size=self.args.test_batch)
        # edge_index = data.edge_index.to(device)
        adj = torch.sparse_coo_tensor(data.edge_index.cpu(), torch.ones(data.edge_index.size(1)), 
                                      size=(data.num_nodes, data.num_nodes))
        adj = utils.normalize(torch.eye(data.num_nodes) + adj).to_dense()

        adj = adj.to(device)
        x = data.x.to(device)
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2)
        criterion = torch.nn.CrossEntropyLoss()

        best_valid_loss = math.inf
        best_epoch = 0
        trial_count = 0
        best_model = None

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
        
        self.model.eval()
        with torch.no_grad():
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

        y_preds, y_true = [], []
        self.model.eval()
        with torch.no_grad():
            for nodes, labels in test_loader:
                nodes = nodes.to(device)
                labels = labels.to(device)
                outputs = self.model(x, adj)
                # outputs = self.model(x, edge_index)
                y_pred = torch.argmax(outputs[nodes], dim=1)
                y_preds.extend(y_pred.cpu().tolist())
                y_true.extend(labels.cpu().tolist())
        
        results = classification_report(y_true, y_preds, digits=4, output_dict=True)
        acc = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_preds, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_preds, average='weighted', zero_division=0)

        return {
            'accuracy': acc,
            'percision': precision,
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


    def parameters(self):
        Ws = [p for p in self.model.parameters() if p.requires_grad]
        # assert len(Ws) == 2, f'Invalid W. There are {len(Ws)} parameters, the sizes: {[w.size() for w in Ws]}'
        if len(Ws) == 2:
            W1, W2 = Ws[0], Ws[1]
            return W1, W2
        else:
            W0, W1, W2, W3 = Ws[0], Ws[1], Ws[2], Ws[3]
            return W0, W1, W2, W3
        

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))