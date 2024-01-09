import os
import copy
import utils
import pickle as pkl
import torch
import torch.nn.functional as F

def preprocess(target, dataset, surrogate, data, node_token, prediction, device):
    init_state_path = os.path.join('explore', f'start_{target}_{dataset}_{node_token}.pkl')
    # if os.path.exists(init_state_path):
    if False:
        with open(init_state_path, 'rb') as fp:
            init_state = pkl.load(fp)
    else:
        surrogate.to(device)
        surrogate.eval()

        ori_adj = data.adjacency_matrix().to(device)
        X = data.x.to(device)
        y = data.y.to(device)

        init_state = {}
        # true_label = y[node_token]
        true_label = torch.tensor(prediction, device=device)

        diff_max = -1000
        for ci in range(data.num_classes):
            if ci == data.y[node_token]:
                continue

            modified_adj = copy.deepcopy(ori_adj.to_dense())
            modified_adj.requires_grad = True

            adj_norm = utils.normalize(modified_adj)
            logits = surrogate(X, adj_norm)             
            output = F.log_softmax(logits, dim=1)
            loss = - (output[node_token, true_label] - output[node_token, ci])
            grad = torch.autograd.grad(loss, modified_adj)[0]
            # bidirection
            grad = (grad[node_token] + grad[:, node_token]) * (-2 * modified_adj[node_token] + 1)
            grad[node_token] = -10
            grad_argmax = torch.argmax(grad)

            value = -2 * modified_adj[node_token][grad_argmax] + 1
            modified_adj.data[node_token][grad_argmax] += value
            modified_adj.data[grad_argmax][node_token] += value

            adj_norm_test = utils.normalize(modified_adj)
            logits = surrogate(X, adj_norm_test)[node_token]   
            # output = F.log_softmax(logits, dim=1)
            diff = logits[ci].item() - logits[true_label].item()
            if diff > diff_max:
                diff_max = diff
                init_state[node_token] = grad_argmax.item()

        # with open(init_state_path, 'wb') as f:
        #     pkl.dump(init_state, f)

    return init_state