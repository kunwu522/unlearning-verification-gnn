"""
Test script for using bash to run python scripts
"""
import torch
import argparse
import data_loader


# def global_value(d_u, d_v):
#     return 2 *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mock')
    parser.add_argument('--model', type=str, default='gcn')

    args = parser.parse_args()

    data = data_loader.load(args.dataset)

    adj = data.adjacency_matrix().to_dense()
    print('adj:', adj)

    adj_tilde = adj + torch.eye(adj.shape[0])
    print('adj_tilde:', adj_tilde)

    degree = torch.sum(adj_tilde, dim=1)
    deg_mx = torch.diag(degree)
    print('degree matrix:', deg_mx)

    deg_mx_inv_sqrt = deg_mx.pow(-0.5)
    adj_norm = torch.mm(torch.mm(deg_mx_inv_sqrt, adj_tilde), deg_mx_inv_sqrt)
    print('adj_norm:', adj_norm)


