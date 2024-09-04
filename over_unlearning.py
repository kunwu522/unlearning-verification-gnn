import os
import time
import copy
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm

import argument
import data_loader
from model.gnn import GNN
import nettack_adapter as ntk

def _second_best_labels(posteriors):
    """ Get the second best label based on the posterior.
    """
    return np.argsort(posteriors, axis=1)[:, -2]

def certify_fragile_with_perturbations(inputs, args, verifier, data):
    target_node, prediction, attack_label = inputs
    # print(mp.current_process().name, 'certifing', target_node, ', prediction:', prediction, 'attack_label:', attack_label, '...')
    if args.candidate_method == 'no':
        candidates = None
    else:
        if args.candidate_method == 'label':
            # Method 1: random sample nodes based on their labels
            candidates = data.train_set.nodes[data.train_set.y == attack_label].tolist()
            candidates = list(set(candidates) - set(data.adj_list[target_node]))
            # print('the number of candidates:', len(candidates))
        elif args.candidate_method == 'random':
            candidates = random.sample(data.train_set.nodes.tolist(), args.candidate_size)
        else:
            raise NotImplementedError('Invalid candidate method:', args.candidate_method)
        # if args.candidate_size > 0 and len(candidates) > args.candidate_size:
        #     candidates = random.sample(candidates, args.candidate_size)
        #     print('candidates:', candidates)

    num_iter = 0
    ps = []
    while len(ps) < args.num_perts:
        if len(candidates) < args.candidate_size:
            break

        if candidates is not None and args.candidate_size > 0:
            _candidates = random.sample(candidates, args.candidate_size)
            # print('candidates:', len(_candidates))
        else:
            _candidates = None

        if args.verbose:
            print('  ==> certifing', target_node, '...')
        t0 = time.time()
        try:
            res = verifier.certify(target_node, prediction, attack_label, candidates=_candidates, solver=args.solver, mask=np.array(ps))
        except Exception as err:
            print('  ==> certifing', target_node, 'error:', err)
            raise err
            # continue
            # return target_node, None

        if (time.time() - t0) > 50:
            print('  ==> certifing', target_node, f', done, in {(time.time() - t0):.2f}s')
            print('step:', res['step'])
            print('fragile_score:', res['fragile_score'])
            print('built_time:', res['buit_time'])
            if not res['fragile']:
                break
        # if args.verbose:
        #     print('  ==> certifing', target_node, f', done, in {(time.time() - t0):.2f}s')
        if res['fragile']:
            p = tuple(res['perturbations'][0])
            if candidates is not None:
                if p[0] == target_node:
                    if p[1] in candidates:
                        candidates.remove(p[1])
                else:
                    if p[0] in candidates:
                        candidates.remove(p[0])

            if p not in ps:
                ps.append(p)
                # print('  ==> certifing', target_node, 'found perturbations,', ps)
        if num_iter == args.T:
            break
        num_iter += 1

    # print('target:', target_node, ': using ', num_iter, 'iterations to find', len(ps), 'ps')
    if len(ps) != args.num_perts:
        return target_node, None, None
    else:
        return target_node, ps, attack_label


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    data = data_loader.load(args)
    target_model = GNN(args, data.num_features, data.num_classes, surrogate=True, bias=False)
    target_model.train(data, device)
    res = target_model.evaluate(data, device)
    print('The accuracy of the target model:', res['accuracy'])

    # sample a subgraph to simulate an authorized user who can access the graph data
    for subgraph_size in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for num_pert in [0.1, 0.2, 0.3, 0.4, 0.5]:
            subgraph_data = copy.deepcopy(data)
            subgraph_data.partial_graph(subgraph_size)

            surrogate = GNN(args, subgraph_data.num_features, subgraph_data.num_classes, surrogate=True, bias=False)
            surrogate.train(subgraph_data, device)

            pertubations = []
            for target_node in tqdm(range(subgraph_data.num_nodes), desc='obtain noise'):
                nettack = ntk.adapte(surrogate, subgraph_data, target_node, add_edge_only=False)
                nettack.attack_surrogate(n_perturbations=int(num_pert * data.num_features), perturb_structure=False, perturb_features=True, direct=True, n_influencers=0)
                pert = [p[1] for p in nettack.feature_perturbations]

                noise = torch.zeros(subgraph_data.num_features, dtype=torch.int, device=device)
                noise[pert] = 1
                pertubations.append(noise)
            
            # Ask the target model to unlearning the perturbed node features
            unlearn_data = copy.deepcopy(data)
            target_nodes = []
            for target_node, noise in enumerate(pertubations):
                orig_idx = subgraph_data.partial_to_original[target_node]
                # unlearn_data.x[orig_idx] = 0
                unlearn_data.x[orig_idx] = torch.bitwise_xor(unlearn_data.x[orig_idx].int(), noise)
                target_nodes.append(orig_idx)
            
            target_model.unlearn(target_nodes, data, unlearn_data, device)
            res = target_model.evaluate(data, device)
            print(f'The over-unlearning model @ {subgraph_size}-{num_pert}:', res['accuracy'])


if __name__ == '__main__':
    parser = argument.load_parser()
    parser.add_argument('--subgraph-size', type=float, default=0.5, help='The size of the subgraph sampled from the original graph')
    args = parser.parse_args()
    main(args)