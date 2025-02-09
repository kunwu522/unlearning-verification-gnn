import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP
from torch_geometric.nn import GINConv

from torch.nn import Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv, global_add_pool

class GIN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, activation=True, bias=True):
        super(GIN, self).__init__()

        linear1 = MLP([nfeat, nhid, nclass], dropout=dropout, bias=bias)
        self.gin = GINConv(linear1, train_eps=True)

        self.dropout = dropout
        self.activation = activation

    def forward(self, x, edge_index):
        return  self.gin(x, edge_index)
        # return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.gin.reset_parameters()
        self.gin.reset_parameters()

# class GIN(torch.nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, activation=True, bias=True, num_layers=3, eps=0.0):
#         super(GIN, self).__init__()
#         self.num_layers = num_layers

#         # Initialize GIN layers
#         self.gin_layers = torch.nn.ModuleList()
#         self.batch_norms = torch.nn.ModuleList()

#         # First layer
#         mlp = torch.nn.Sequential(
#             Linear(nfeat, nhid),
#             ReLU(),
#             Linear(nhid, nclass)
#         )
#         self.gin_layers.append(GINConv(mlp, train_eps=True))
#         self.batch_norms.append(BatchNorm1d(nclass))

#         # Hidden layers
#         for _ in range(num_layers - 1):
#             mlp = torch.nn.Sequential(
#                 Linear(nhid, nhid),
#                 ReLU(),
#                 Linear(nhid, nhid)
#             )
#             self.gin_layers.append(GINConv(mlp, train_eps=True))
#             self.batch_norms.append(BatchNorm1d(nhid))

#         # Final linear layer for node-level prediction
#         self.linear_out = Linear(nhid, nclass)

#     def forward(self, x, edge_index):
#         # Apply GIN layers
#         for layer, batch_norm in zip(self.gin_layers, self.batch_norms):
#             x = layer(x, edge_index)
#             x = batch_norm(x)
#             x = torch.relu(x)