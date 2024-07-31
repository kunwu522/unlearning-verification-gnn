import numpy as np
import scipy.sparse as sp
from nettack import nettack as ntk


def adapte(surrogate, data, target_node, prediction=None, target_label=None, add_edge_only=True, epsilon=0.): 
    # contruct adjacency matrix
    row = data.edge_index.numpy()[0]
    col = data.edge_index.numpy()[1]
    value = np.ones((len(row)))
    adj = sp.csr_matrix((value, (row, col)), shape=(data.num_nodes, data.num_nodes))

    # contruct features sparse
    row, col = np.where(data.x.numpy() != 0)
    value = data.x.numpy()[row, col]
    x = sp.csr_matrix((value, (row, col)), shape=data.x.shape)
    labels = data.y.numpy()

    W1, W2 = surrogate.parameters()
    W1 = W1.detach().cpu().numpy()
    W2 = W2.detach().cpu().numpy()

    _nettack = ntk.Nettack(adj, x, labels, data.num_classes, W1, W2, target_node, 
                            add_edge_only=add_edge_only,
                            target_prediction=prediction,
                            target_label=target_label,
                            verbose=False, epsilon=epsilon) 
    return _nettack