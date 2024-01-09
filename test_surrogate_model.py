import copy
import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
from tqdm import tqdm
import argument
import data_loader
from model.gcn import GNN
from nettack import nettack as ntk
import utils

if __name__ == '__main__':
    parser = argument.load_parser()
    args = parser.parse_args()

    data = data_loader.load(args)
    device = utils.get_device(args)

    surrogate = GNN(args, data.num_features, data.num_classes, surrogate=True)
    surrogate.train(data, device)

    clean_model = GNN(args, data.num_features, data.num_classes, surrogate=False)
    clean_model.train(data, device)

    _, _, surr_posterior = surrogate.predict(data, device, return_posterior=True)
    _, _, clean_posterior = clean_model.predict(data, device, return_posterior=True)

    surr_node2boundary_scores = {}
    for idx, p in enumerate(surr_posterior):
        surr_node2boundary_scores[idx] = utils.boundary_score(p)
    
    clean_node2boundary_scores = {}
    for idx, p in enumerate(clean_posterior):
        clean_node2boundary_scores[idx] = utils.boundary_score(p)
    
    sorted_surr = {k: v for k, v in sorted(surr_node2boundary_scores.items(), key=lambda item: item[1])}
    sorted_clean = {k: v for k, v in sorted(clean_node2boundary_scores.items(), key=lambda item: item[1])}

    surr_top_boundary_nodes = list(sorted_surr.keys())[:100]
    clean_top_boundary_nodes = list(sorted_clean.keys())[:100]

    result = len(set(surr_top_boundary_nodes).intersection(set(clean_top_boundary_nodes)))
    print('The overlap is', result)

    # Nettack
    W1, W2 = surrogate.parameters()
    W1 = W1.detach().numpy().T
    W2 = W2.detach().numpy().T
    row = data.edge_index.numpy()[0]
    col = data.edge_index.numpy()[1]
    value = np.ones((len(row)))
    adj = sp.csr_matrix((value, (row, col)), shape=(data.num_nodes, data.num_nodes))
    row, col = np.where(data.x.numpy() == 1)
    value = np.ones((len(row)))
    x = sp.csr_matrix((value, (row, col)), shape=data.x.shape)
    labels = data.y.numpy()

    tmp_clean_pred, tmp_adv_pred = [], []
    target_nodes = set(surr_top_boundary_nodes).intersection(set(clean_top_boundary_nodes))
    for idx, target_node in enumerate(tqdm(target_nodes, desc='nettacking')):
        direct_attack = True
        n_influencers = 1 if direct_attack else 5
        n_perturbations = int(data.degree(target_node)) * 2
        perturb_features = False
        perturb_structure = True
        target_preds, _, = surrogate.predict(data, device, target_nodes=[target_node])

        _nettack = ntk.Nettack(adj, x, labels, W1, W2, target_node, 
                                add_edge_only=True,
                                target_prediction=target_preds[0])
        # _nettack = ntk.Nettack(adj, x, labels, W1, W2, target_node, 
        #                        add_edge_only=args.add_edge_only)
        _nettack.reset()
        _nettack.attack_surrogate(n_perturbations, 
                                    perturb_structure=perturb_structure,
                                    perturb_features=perturb_features,
                                    direct=direct_attack,
                                    n_influencers=n_influencers)
        
        perturbed_edges = np.array(np.where(_nettack.adj.toarray() != adj.toarray()))
        # node2adv_edges[target_node] = perturbed_edges.T.tolist()
        num_perturbed_edges = int(perturbed_edges.shape[1] / 2)

        original_connection = adj.toarray()[perturbed_edges[0], perturbed_edges[1]][:num_perturbed_edges]
        num_adding = np.sum(original_connection == 0).item()
        num_removing = np.sum(original_connection == 1).item()

        if args.verbose:
            print('*' * 5, 'Adversarial Edge', '*' * 5)
            print('# of perturbed edges:', num_perturbed_edges)
            print('perturbed edges:', perturbed_edges.T.tolist())
            print('# of adding', num_adding)
            print('# of removing', num_removing)
            print('=' * 20)

        '''
        Evaluate the attack one target node at a time.
        '''
        adv_data = copy.deepcopy(data)
        adv_data.add_edges(torch.from_numpy(perturbed_edges))
        # train clean model
        clean_model = GNN(args, data.num_features, data.num_classes, surrogate=False)
        clean_model.train(data, device)
        clean_pred, true_label, clean_post = clean_model.predict(data, device, target_nodes=[target_node], return_posterior=True)

        adv_model = GNN(args, adv_data.num_features, adv_data.num_classes, surrogate=False)
        adv_model.train(adv_data, device)
        adv_pred, _, adv_post = adv_model.predict(adv_data, device, target_nodes=[target_node], return_posterior=True)

        tmp_clean_pred.extend(clean_pred)
        tmp_adv_pred.extend(adv_pred)
        # result['target node'].append(target_node)
        # result['num adv edges'].append(num_perturbed_edges)
        # result['num adding edges'].append(num_adding)
        # result['num removing edges'].append(num_removing)
        # result['adv edges'].append(perturbed_edges.T.tolist())
        # result['true label'].append(true_label)
        # result['clean prediction'].extend(clean_pred)
        # result['adv prediction'].extend(adv_pred)
        # result['clean posterior'].append(clean_post)
        # result['adv posterior'].append(adv_post)
        # result['boundary score'].append(boundary_scores[idx])
    
    asr = np.sum(np.array(tmp_clean_pred) != np.array(tmp_adv_pred)) / len(tmp_clean_pred)
    print(f'The current ASR {asr:.4f}')