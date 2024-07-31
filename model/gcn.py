import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, activation=True, bias=True):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, bias=bias)
        self.gc2 = GraphConvolution(nhid, nclass, bias=bias)
        self.dropout = dropout
        self.activation = activation

    def forward(self, x, adj):
        if self.activation:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
        # return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
    


class GCN3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, activation=True, bias=True):
        super(GCN3, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid[0], bias=bias)
        self.gc2 = GraphConvolution(nhid[0], nhid[1], bias=bias)
        self.gc3 = GraphConvolution(nhid[1], nclass, bias=bias)
        self.dropout = dropout
        self.activation = activation

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return x
        # return F.log_softmax(x, dim=1)

class GCN1(nn.Module):
    def __init__(self, nfeat, nclass, bias=True):
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nclass, bias=bias)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        return x