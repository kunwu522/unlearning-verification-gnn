import numpy as np
import scipy as sp
# import scipy.sparse as sp
from GF_Attack import GFA
import GF_Attack.utils as gf_utils

def adapte(data, target_node, n_perturbations, device):
    row = data.edge_index.numpy()[0]
    col = data.edge_index.numpy()[1]
    value = np.ones((len(row)))
    adj = sp.sparse.csr_matrix((value, (row, col)), shape=(data.num_nodes, data.num_nodes))

    # contruct features sparse
    row, col = np.where(data.x.numpy() == 1)
    value = np.ones((len(row)))
    x = sp.sparse.csr_matrix((value, (row, col)), shape=data.x.shape)
    labels = data.y.numpy()

    X_mean = np.sum(x, axis=1)
    K = 2
    T = int(data.num_nodes / 2)

    _An = gf_utils.preprocess_graph(adj)
    A_processed = _An
    A_I = adj + sp.sparse.eye(data.num_nodes)
    A_I[A_I > 1] = 1
    rowsum = A_I.sum(1).A1

    degree_mat = sp.sparse.diags(rowsum)

    eig_vals, eig_vec = sp.linalg.eigh(A_I.todense(), degree_mat.todense())

    gf_attack = GFA.GFA(adj, labels, target_node, X_mean, eig_vals, eig_vec, K, T, None)
    gf_attack.reset()
    gf_attack.attack_model(n_perturbations)

    return gf_attack