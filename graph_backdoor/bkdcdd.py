import os
import sys
import copy
import random
import numpy as np
from itertools import permutations
from collections import defaultdict
import torch
from torch_geometric.loader import DataLoader
# from utils.datareader import DataReader
# sys.path.append('/home/zxx5113/BackdoorGNN/')


def filter_edges(edge_index, nodes):
    node_indices = np.isin(edge_index, nodes)
    _edge_index = edge_index.T[np.where(~np.all(node_indices, axis=0))]
    return torch.from_numpy(_edge_index.T)


def backdoor_data(data, bkd_size):
    benign_labels = []
    benign_loader = DataLoader(data.train_set, batch_size=64)
    for batch in benign_loader:
        benign_labels.append(batch.y)
    benign_labels = torch.cat(benign_labels)
    print('# of nodes:', benign_labels.size())
    print('# of label 1:', torch.sum(benign_labels))

    bkd_data = copy.deepcopy(data)
    v, c = torch.unique(bkd_data.train_labels, return_counts=True)
    target_label = v[torch.argsort(c)[0]].reshape(1)

    num_train_bkd_graphs = int(len(bkd_data.train_set) * 0.1)
    num_test_bkd_graphs = int(len(bkd_data.test_set) * 0.5)

    bkd_data.bkd_train_graphs = []
    sampled_indices = []
    changed_labels = defaultdict(list)
    while len(bkd_data.bkd_train_graphs) < num_train_bkd_graphs:
        idx = random.choice(range(len(bkd_data.train_set)))
        if idx in sampled_indices:
            continue

        graph = bkd_data.train_set[idx]
        changed_labels[graph.y.item()].append(graph.num_nodes)
        if graph.num_nodes > 1 * bkd_size:
            bkd_nodes = random.sample(list(range(graph.num_nodes)), bkd_size)
            print('@@@@', bkd_nodes)
            complete_edge_index = torch.tensor(list(permutations(bkd_nodes, 2))).t()
            _edge_index = filter_edges(graph.edge_index.numpy(), np.array(bkd_nodes))
            _edge_index = torch.cat((_edge_index, complete_edge_index), dim=1)
            graph.edge_index = _edge_index
            graph.y = target_label
            bkd_data.bkd_train_graphs.append(graph)
            bkd_data.train_set[idx] = graph
            sampled_indices.append(idx)
    # for idx in sampled_indices:
    #     print(bkd_data.train_set[idx].edge_index.size(), data.train_set[idx].edge_index.size())
    #     print(bkd_data.train_set[idx].y, data.train_set[idx].y)
    # print(len(sampled_indices))
    print('changed:', changed_labels)

    bkd_labels = []
    bkd_loader = DataLoader(bkd_data.train_set, batch_size=64)
    for batch in bkd_loader:
        bkd_labels.append(batch.y)
    bkd_labels = torch.cat(bkd_labels)
    print('# of label 1:', torch.sum(bkd_labels))

    print(benign_labels[sampled_indices], benign_labels.size())
    print(bkd_labels[sampled_indices], bkd_labels.size())

    # print(np.unique(y, return_counts=True))
    # print(np.unique(bkd_y, return_counts=True))

    bkd_data.bkd_test_graphs = []
    sampled_indices = []
    while len(bkd_data.bkd_test_graphs) < num_test_bkd_graphs:
        idx = random.choice(range(len(bkd_data.train_set)))
        if idx in sampled_indices:
            continue

        graph = bkd_data.test_set[idx]
        if graph.num_nodes > 1 * bkd_size:
            bkd_nodes = random.sample(list(range(graph.num_nodes)), bkd_size)
            complete_edge_index = torch.tensor(list(permutations(bkd_nodes, 2))).t()
            _edge_index = filter_edges(graph.edge_index.numpy(), np.array(bkd_nodes))
            _edge_index = torch.cat((_edge_index, complete_edge_index), dim=1)
            graph.edge_index = _edge_index
            graph.y = target_label
            bkd_data.bkd_test_graphs.append(graph)
            bkd_data.test_set[idx] = graph
            sampled_indices.append(idx)

    return bkd_data


# return 1D list
def select_cdd_graphs(args, data, subset: str):
    '''
    Given a data (train/test), (randomly or determinately) 
    pick up some graph to put backdoor information, return ids.
    '''
    rs = np.random.RandomState(args.seed)
    graph_sizes = [d.num_nodes for d in data.dataset]
    bkd_graph_ratio = args.bkd_gratio_train if subset == 'train' else args.bkd_gratio_test
    bkd_num = int(np.ceil(bkd_graph_ratio * len(data.dataset)))
    
    assert len(data)>bkd_num , "Graph Instances are not enough"
    picked_ids = []
    
    # Randomly pick up graphs as backdoor candidates from data
    remained_set = copy.deepcopy(data.dataset)
    loopcount = 0
    while bkd_num-len(picked_ids) >0 and len(remained_set)>0 and loopcount<=50:
        loopcount += 1
        
        cdd_ids = rs.choice(remained_set, bkd_num-len(picked_ids), replace=False)
        for gid in cdd_ids:
            if bkd_num-len(picked_ids) <=0: 
                break
            gsize = graph_sizes[gid]
            if gsize >= 3*args.bkd_size*args.bkd_num_pergraph:
                picked_ids.append(gid)

        if len(remained_set)<len(data):
            for gid in cdd_ids:
                if bkd_num-len(picked_ids) <=0: 
                    break
                gsize = graph_sizes[gid]
            if gsize >= 1.5*args.bkd_size*args.bkd_num_pergraph and gid not in picked_ids:
                    picked_ids.append(gid)
                    
        if len(remained_set)<len(data):
            for gid in cdd_ids:
                if bkd_num-len(picked_ids) <=0: 
                    break
                gsize = graph_sizes[gid]
                if gsize >= 1.0*args.bkd_size*args.bkd_num_pergraph and gid not in picked_ids:
                    picked_ids.append(gid)
                    
        picked_ids = list(set(picked_ids))
        remained_set = list(set(remained_set) - set(picked_ids))
        if len(remained_set)==0 and bkd_num>len(picked_ids):
            print("no more graph to pick, return insufficient candidate graphs, try smaller bkd-pattern or graph size")

    return picked_ids
             

def select_cdd_nodes(args, graph_cdd_ids, adj_list):
    '''
    Given a graph instance, based on pre-determined standard,
    find nodes who should be put backdoor information, return
    their ids.

    return: same sequece with bkd-gids
            (1) a 2D list - bkd nodes under each graph
            (2) and a 3D list - bkd node groups under each graph
                (in case of each graph has multiple triggers)
    '''
    rs = np.random.RandomState(args.seed)
    
    # step1: find backdoor nodes
    picked_nodes = []  # 2D, save all cdd graphs
    
    for gid in graph_cdd_ids:
        node_ids = [i for i in range(len(adj_list[gid]))]
        assert len(node_ids)==len(adj_list[gid]), 'node number in graph {} mismatch'.format(gid)

        bkd_node_num =  int(args.bkd_num_pergraph*args.bkd_size)
        assert bkd_node_num <= len(adj_list[gid]), "error in SelectCddGraphs, candidate graph too small"
        cur_picked_nodes = rs.choice(node_ids, bkd_node_num, replace=False)
        picked_nodes.append(cur_picked_nodes)
        
    # step2: match nodes
    assert len(picked_nodes)==len(graph_cdd_ids), "backdoor graphs & node groups mismatch, check SelectCddGraphs/SelectCddNodes"

    node_groups = [] # 3D, grouped trigger nodes
    for i in range(len(graph_cdd_ids)):    # for each graph, devide candidate nodes into groups
        gid = graph_cdd_ids[i]
        nids = picked_nodes[i]

        assert len(nids)%args.bkd_size==0.0, "Backdoor nodes cannot equally be divided, check SelectCddNodes-STEP1"

        # groups within each graph
        groups = np.array_split(nids, len(nids)//args.bkd_size)
        # np.array_split return list[array([..]), array([...]), ]
        # thus transfer internal np.array into list
        # store groups as a 2D list.
        groups = np.array(groups).tolist()
        node_groups.append(groups)

    assert len(picked_nodes)==len(node_groups), "groups of bkd-nodes mismatch, check SelectCddNodes-STEP2"
    return picked_nodes, node_groups
                           
    