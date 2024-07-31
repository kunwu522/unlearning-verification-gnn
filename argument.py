'''
Common arguments for training GNN models

Copyright (C) 2023
Kun Wu
Stevens Institute of Technology
'''
import argparse

def load_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num-trials', dest='num_trials', type=int, default=5)
    parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='cora', help='The name of datasets, cora|citeseer|pubmed')
    parser.add_argument('--target', type=str, default='gcn', help='Target model, gcn|gat')
    parser.add_argument('--hidden-size', dest='hidden_size', type=int, default=16)
    parser.add_argument('--epochs', dest='epochs', type=int, default=500)
    parser.add_argument('--patience', dest='patience', type=int, default=20)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--l2', dest='l2', type=float, default=1e-5)
    parser.add_argument('--batch', dest='batch', type=int, default=512)
    parser.add_argument('--test-batch', dest='test_batch', type=int, default=1024)
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--subgraph', dest='subgraph', type=int, default=None)
    parser.add_argument('--solver', dest='solver', type=str, default='cplex', help='Solver for the convex problem, ECOS|OSQP|SCS')
    return parser