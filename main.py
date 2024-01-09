import os
import copy
import time
import random
import numpy as np
import scipy.sparse as sp
import pandas as pd
import data_loader
import torch
from tqdm import tqdm
from collections import defaultdict
import argument
from model.gcn import GNN
from nettack import nettack as ntk
import utils

def sample_target_nodes(args, data, candidate_nodes, posterior=None):
    if args.sampler == 'random':
        result = random.sample(candidate_nodes, min(args.num_target_nodes, len(candidate_nodes)))
    elif args.sampler == 'boundary' or args.sampler == 'distance':
        near_boundary_nodes = {}
        for idx, p in enumerate(posterior):
            # if utils.near_boundary(z, args.k):
            near_boundary_nodes[candidate_nodes[idx]] = utils.boundary_score(p)
        sorted_boundary_nodes = {k: v for k,v in sorted(near_boundary_nodes.items(), key=lambda item: item[1], reverse=args.sampler == 'distance')}
        boundary_scores = list(sorted_boundary_nodes.values())[:min(len(candidate_nodes), args.num_target_nodes)]
        
        # i = 0
        # result = []
        # while len(result) < args.num_target_nodes:
        #     n = list(sorted_boundary_nodes.keys())[i]
        #     if data.degree(n) != 0:
        #         result.append(n)
        #     i += 1
        result = list(sorted_boundary_nodes.keys())[:min(len(candidate_nodes), args.num_target_nodes)]
    elif args.sampler == 'mind' or args.sampler == 'maxd':
        node_degree = {}
        for node in candidate_nodes:
            node_degree[node] = data.degree(node)
        sorted_node_degree = {k: v for k, v in sorted(
            node_degree.items(), key=lambda item: item[1], reverse=args.sampler == 'maxd')}
        result = list(sorted_node_degree.keys())[:min(len(candidate_nodes), args.num_target_nodes)]

    if args.sampler == 'boundary' or args.sampler == 'distance':
        return result, boundary_scores
    else:
        return result 


def adversarial_attack(args):
    if torch.has_mps:
        device = 'mps'
    else:
        device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')

    result = defaultdict(list)
    for _ in range(args.num_trials):
        # re-create data in order to resplit train/test set.
        data = data_loader.load(args)

        surrogate = GNN(args, data.num_features, data.num_classes, surrogate=True)
        surrogate.train(data, device=device)
        W1, W2 = surrogate.parameters()
        W1 = W1.detach().numpy().T
        W2 = W2.detach().numpy().T

        # Nettack
        row = data.edge_index.numpy()[0]
        col = data.edge_index.numpy()[1]
        value = np.ones((len(row)))
        adj = sp.csr_matrix((value, (row, col)), shape=(data.num_nodes, data.num_nodes))
        row, col = np.where(data.x.numpy() == 1)
        value = np.ones((len(row)))
        x = sp.csr_matrix((value, (row, col)), shape=data.x.shape)
        labels = data.y.numpy()

        candidate_target_nodes = data.train_set.nodes.tolist() if args.target_train else data.test_set.nodes.tolist()
        if args.sampler == 'boundary' or args.sampler == 'distance':
            _, _, posterior, Z = surrogate.predict(data, device, target_nodes=candidate_target_nodes, return_posterior=True, return_logit=True)
            target_nodes, boundary_scores = sample_target_nodes(args, data, candidate_target_nodes, posterior)
            # target_nodes, boundary_scores = sample_target_nodes(args, data, candidate_target_nodes, Z)

            # v_idx = candidate_target_nodes.index(target_nodes[0])
            # print(posterior[v_idx])
            # print(Z[v_idx])
            # u_idx = candidate_target_nodes.index(target_nodes[1])
            # print(posterior[u_idx])
            # print(Z[u_idx])
            # exit(0)
        else:
            target_nodes = sample_target_nodes(args, data, candidate_target_nodes)
            _, _, posterior = surrogate.predict(data, device, target_nodes=target_nodes, return_posterior=True)
            boundary_scores = []
            for p in posterior:
                boundary_scores.append(utils.boundary_score(p))

        # _pred, _truth, _posterior = surrogate.predict(data, device, target_nodes=target_nodes, return_posterior=True)
        # print('-' * 10 + ' Target nodes ' + '-' * 10)
        # print(target_nodes)
        # print(_posterior.tolist())
        # print(_pred)
        # print(_truth)
        # print('=' * 34)


        tmp_clean_pred, tmp_adv_pred = [], []
        adv_edges = None
        node2adv_edges = {}
        # for idx, target_node in enumerate(target_nodes):
        for idx, target_node in enumerate(tqdm(target_nodes, desc='nettacking')):
            direct_attack = args.direct_attack
            n_influencers = 1 if direct_attack else 5
            n_perturbations = int(data.degree(target_node)) * 2 if data.degree(target_node) != 0 else 2
            perturb_features = False
            perturb_structure = True
            surrogate_preds, surrogate_truth, surrogate_post= surrogate.predict(data, device, target_nodes=[target_node], return_posterior=True)

            # if n_perturbations == 0:
            #     continue

            _nettack = ntk.Nettack(adj, x, labels, W1, W2, target_node, 
                                   add_edge_only=args.add_edge_only,
                                   target_prediction=surrogate_preds[0],
                                   verbose=False)
            # _nettack = ntk.Nettack(adj, x, labels, W1, W2, target_node, 
            #                        add_edge_only=args.add_edge_only)
            _nettack.reset()
            _nettack.attack_surrogate(n_perturbations, 
                                      perturb_structure=perturb_structure,
                                      perturb_features=perturb_features,
                                      direct=direct_attack,
                                      n_influencers=n_influencers)
            adv_nodes = set()
            for u, v in _nettack.structure_perturbations:
                if u != target_node:
                    adv_nodes.add(u)
                if v != target_node:
                    adv_nodes.add(v)
            adv_nodes = list(adv_nodes)
            perturbed_edges = np.array(np.where(_nettack.adj.toarray() != adj.toarray()))
            node2adv_edges[target_node] = perturbed_edges.T.tolist()
            if adv_edges is None:
                adv_edges = perturbed_edges
            else:
                adv_edges = np.concatenate((adv_edges, perturbed_edges), axis=1)
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
            result['target node'].append(target_node)
            result['num adv edges'].append(num_perturbed_edges)
            result['num adding edges'].append(num_adding)
            result['num removing edges'].append(num_removing)
            result['adv edges'].append(perturbed_edges.T.tolist())
            result['true label'].append(true_label)
            result['clean prediction'].extend(clean_pred)
            result['adv prediction'].extend(adv_pred)
            result['clean posterior'].append(clean_post)
            result['adv posterior'].append(adv_post)
            result['boundary score'].append(boundary_scores[idx])

            if clean_pred[0] == adv_pred[0]:
                print('#' * 15, f'Target node: {target_node}', '#' * 15)
                print('            ground truth:', data.y[target_node].item())
                print('    surrogate prediction:', surrogate_preds[0])
                print('        clean prediction:', clean_pred[0])
                print('  adversarial prediction:', adv_pred[0])
                print('    surrogate post order:', np.argsort(surrogate_post.squeeze())[::-1])
                print('        clean post order:', np.argsort(clean_post.squeeze())[::-1])
                print('  adversarial post order:', np.argsort(adv_post.squeeze())[::-1])
                print('       adversarial nodes:', adv_nodes)
                print('adversarial nodes labels:', data.y[adv_nodes].tolist())
                print('#' * 50)

        '''
        Evaluate the attack by adding adv edges by once
        '''
        # adv_data = copy.deepcopy(data)
        # adv_data.add_edges(torch.from_numpy(adv_edges))
        # print('The number of adv edges:', adv_edges.shape)
        # print(adv_edges)
        # common_nodes = set(target_nodes).intersection(set(adv_edges[0].tolist()))
        # common_nodes = common_nodes.intersection(set(adv_edges[1].tolist()))
        # print('The number of common nodes:', len(common_nodes))
        # print()
    
        # # train clean model
        # clean_model = GNN(args, data.num_features, data.num_classes, surrogate=False)
        # clean_model.train(data, device)
        # clean_pred, true_label, clean_post = clean_model.predict(data, device, target_nodes=target_nodes, return_posterior=True)

        # adv_model = GNN(args, adv_data.num_features, adv_data.num_classes, surrogate=False)
        # adv_model.train(adv_data, device)
        # adv_pred, _, adv_post = adv_model.predict(adv_data, device, target_nodes=target_nodes, return_posterior=True)

        # tmp_clean_pred.extend(clean_pred)
        # tmp_adv_pred.extend(adv_pred)
        # result['target node'].extend(target_nodes)
        # result['num adv edges'].extend(list(node2adv_edges.values()))
        # # result['num adding edges'].append(num_adding)
        # # result['num removing edges'].append(num_removing)
        # # result['adv edges'].append(perturbed_edges.T.tolist())
        # result['true label'].extend(data.y[target_nodes].tolist())
        # result['clean prediction'].extend(clean_pred)
        # result['adv prediction'].extend(adv_pred)
        # # result['clean posterior'].append(clean_post)
        # # result['adv posterior'].append(adv_post)
        # # if args.sampler == 'boundary':
        # result['boundary score'].extend(boundary_scores)

        asr = np.sum(np.array(tmp_clean_pred) != np.array(tmp_adv_pred)) / len(tmp_clean_pred)
        print(f'The current ASR {asr:.4f}')

    df = pd.DataFrame(data=result)
    # calculate and print out the attack accuracy
    clean_pred = df['clean prediction'].values
    adv_pred = df['adv prediction'].values
    attack_acc = np.sum(clean_pred != adv_pred) / len(clean_pred)
    print(f'Attack accuracy: {attack_acc:.4f}')

    result_file_path = 'nettack'
    result_file_path += '_train' if args.target_train else '_test' 
    result_file_path += f'_{args.sampler}'
    if args.add_edge_only:
        result_file_path += '_edge-only'
    timestamp = int(time.time())
    result_file_path += f'_{timestamp}.csv'
    df.to_csv(os.path.join('./result', result_file_path))
    print(f'==> Experiment finished, save the result in {result_file_path}')


def verify(args):
    device = utils.get_device(args)

    adv_predictions = []
    clean_predictions = []

    for _ in tqdm(range(args.num_trials)):
        data = data_loader.load(args)
        surrogate = GNN(args, data.num_features, data.num_classes, surrogate=True)
        surrogate.train(data, device=device)
        W1, W2 = surrogate.parameters()
        W1 = W1.detach().numpy().T
        W2 = W2.detach().numpy().T

        removed_edge_index, remain_edge_index = utils.sample_edges(args, data.edge_index, 100)
        connected_nodes = removed_edge_index.flatten()
        
        num_target_nodes = 1
        target_nodes = random.sample(connected_nodes.tolist(), num_target_nodes)
 
        # Nettack
        row = data.edge_index.numpy()[0]
        col = data.edge_index.numpy()[1]
        value = np.ones((len(row)))
        adj = sp.csr_matrix((value, (row, col)), shape=(data.num_nodes, data.num_nodes))

        row = remain_edge_index.edge_index.numpy()[0]
        col = remain_edge_index.edge_index.numpy()[1]
        value = np.ones((len(row)))
        adj_ul = sp.csr_matrix((value, (row, col)), shape=(data.num_nodes, data.num_nodes))

        row, col = np.where(data.x.numpy() == 1)
        value = np.ones((len(row)))
        x = sp.csr_matrix((value, (row, col)), shape=data.x.shape)
        labels = data.y.numpy()

        for target_node in target_nodes:
            direct_attack = args.direct_attack
            n_influencers = 1 if direct_attack else 5
            n_perturbations = int(data.degree(target_node)) * 2
            perturb_features = False
            perturb_structure = True
            surrogate_preds, surrogate_truth, surrogate_post= surrogate.predict(data, device, target_nodes=[target_node], return_posterior=True)

            _nettack = ntk.Nettack(adj, x, labels, W1, W2, target_node, 
                                    add_edge_only=args.add_edge_only,
                                    target_prediction=surrogate_preds[0],
                                    verbose=False, adj_ul=adj_ul)
            _nettack.reset()
            _nettack.attack_surrogate(n_perturbations, 
                                        perturb_structure=perturb_structure,
                                        perturb_features=perturb_features,
                                        direct=direct_attack,
                                        n_influencers=n_influencers)
            # adv_nodes = set()
            # for u, v in _nettack.structure_perturbations:
            #     if u != target_node:
            #         adv_nodes.add(u)
            #     if v != target_node:
            #         adv_nodes.add(v)
            # adv_nodes = list(adv_nodes)
            perturbed_edges = np.array(np.where(_nettack.adj.toarray() != adj.toarray()))
            # if adv_edges is None:
            #     adv_edges = perturbed_edges
            # else:
            #     adv_edges = np.concatenate((adv_edges, perturbed_edges), axis=1)
            # num_perturbed_edges = int(perturbed_edges.shape[1] / 2)
            
            '''
            Evaluate the attack one target node at a time.
            '''
            adv_data = copy.deepcopy(data)
            adv_data.add_edges(torch.from_numpy(perturbed_edges))
            adv_model = GNN(args, adv_data.num_features, adv_data.num_classes, surrogate=False)
            adv_model.train(adv_data, device)
            adv_pred, _, adv_post = adv_model.predict(adv_data, device, target_nodes=[target_node], return_posterior=True)

            # train clean model
            retrain_data = copy.deepcopy(data)
            retrain_data.edge_index = remain_edge_index
            clean_model = GNN(args, retrain_data.num_features, retrain_data.num_classes, surrogate=False)
            clean_model.train(retrain_data, device)
            clean_pred, true_label, clean_post = clean_model.predict(retrain_data, device, target_nodes=[target_node], return_posterior=True)

            adv_predictions.append(adv_pred)
            clean_predictions.append(clean_pred)

    print(adv_predictions)
    print(clean_predictions)
    p = np.sum(np.concatenate(adv_predictions) == np.concatenate(clean_predictions))
    print('The p is', p / args.num_trials)



if __name__ == '__main__':
    parser = argument.load_parser()

    # for nettack
    parser.add_argument('--num-target-nodes', dest='num_target_nodes', type=int, default=100)
    parser.add_argument('--sampler', dest='sampler', type=str, default='random', 
                        help='How to sample target nodes, random|boundary|maxd|mind|distance')
    parser.add_argument('--method', dest='method', type=str, default='random', 
                        help='How to sample edges to forget, random')
    parser.add_argument('-k', type=float, default=0.01, 
                        help='Refect how a node close to the classification boundary')
    parser.add_argument('--add-edge-only', dest='add_edge_only', action='store_true')
    parser.add_argument('--target-train', action='store_true')
    parser.add_argument('--no-direct-attack',dest='direct_attack', action='store_false')

    args = parser.parse_args()

    adversarial_attack(args)

    # verify(args)
