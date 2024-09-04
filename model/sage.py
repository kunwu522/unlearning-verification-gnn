import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, activation=True, bias=True):
        super(GraphSAGE, self).__init__()

        self.sage1 = SAGEConv(nfeat, nhid, bias=bias)
        self.sage2 = SAGEConv(nhid, nclass, bias=bias)
        self.dropout = dropout
        self.activation = activation

    def forward(self, x, edge_index):
        if self.activation:
            x = F.relu(self.sage1(x, edge_index))
        else:
            x = self.sage1(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        return x
        # return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.sage1.reset_parameters()
        self.sage2.reset_parameters()