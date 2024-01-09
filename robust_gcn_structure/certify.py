import math
import numpy as np
import scipy.sparse as sp
import torch
from .certification import certify



def is_1perturbation_fragile_node(model, data, v, prediction, 
                                  local_budget=1, global_budget=1,
                                  max_iters=100, tolerance=1e-2,
                                  solver='ECOS'):
    """
        model: GNN model, normally is a surrogate model.
        v: the target node to certify.
        data: the dataset.
    """
    # get parameters
    params = [p.cpu().detach().numpy() for p in model.parameters()]

    # contruct adjacency matrix
    row = data.edge_index.numpy()[0]
    col = data.edge_index.numpy()[1]
    value = np.ones((len(row)))
    adj = sp.csr_matrix((value, (row, col)), shape=(data.num_nodes, data.num_nodes))
    
    X = data.x.numpy()
    y = data.y.numpy()
    
    # class2fragile = {}
    # most_fragile_class = None
    best_upper = math.inf
    non_robust_nodes = []

    for c in range(data.num_classes):
        if c == prediction:
            continue
        results = certify(v, adj, X, params, y,
                          local_changes=local_budget,
                          global_changes=global_budget,
                          solver=solver, eval_class=c, use_predicted_class=True,
                          max_iter=max_iters, tolerance=tolerance)
        if 'robust' not in results:
            continue
        if not results['robust']:
            return True
        # class2fragile[c] = not results['robust']
        # class2fragile[c] = results['fragile']
        # if results['fragile'] and results['best_upper'] < best_upper:
        #     best_upper = results['best_upper']
        #     most_fragile_class = c
    
    return False