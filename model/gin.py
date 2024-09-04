import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP
from torch_geometric.nn import GINConv

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