import random
import pickle
from collections import defaultdict, namedtuple
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import networkx as nx
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Coauthor, FacebookPagePage, CitationFull, Amazon, LastFMAsia, PolBlogs
from ogb.lsc import MAG240MDataset
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.utils import subgraph, to_torch_coo_tensor, to_edge_index, k_hop_subgraph, is_undirected, to_undirected, to_networkx, homophily
from sklearn.model_selection import train_test_split

import argument
import utils

class TmpData:
    def __init__(self, x, y, edge_index):
        self.x = x
        self.y = y
        self.edge_index = edge_index

def load_single_ego_network():
    ego_id =107
    label_name_prefix = 'locale'

    nodes, edges = set(), set()
    with open(f'./data/facebook/{ego_id}.edges', 'r') as fp:
        for line in fp:
            splited = line.strip().split(' ')
            assert len(splited) == 2, f'Invalid edge: {line}.'
            edges.add(tuple(map(int, splited)))
            nodes.add(int(splited[0]))
            nodes.add(int(splited[1]))
    for v in nodes:
        edges.add((ego_id, v))
    nodes.add(ego_id)

    nodes, edges = sorted(list(nodes)), list(edges)
    node2idx = {node: i for i, node in enumerate(nodes)}
    edges = torch.tensor([[node2idx[edge[0]], node2idx[edge[1]]] for edge in edges], dtype=torch.long).t()

    feat_names = set()
    label_names = set()
    label_indces = []
    with open(f'./data/facebook/{ego_id}.featnames', 'r') as fp:
        for i, line in enumerate(fp):
            splited = line.strip().split(' ')
            feat_name = ' '.join(splited[1:])
            if feat_name.startswith(label_name_prefix):
                label_names.add(feat_name)
                label_indces.append(i)
            else:
                feat_names.add(feat_name)
    feat_names = list(feat_names)
    label_names = list(label_names)
        
    dropped_nodes = []
    features = torch.zeros(len(nodes), len(feat_names), dtype=torch.float)
    labels = torch.zeros(len(nodes), len(label_names), dtype=torch.int)
    with open(f'./data/facebook/{ego_id}.feat', 'r') as fp:
        for line in fp:
            splited = line.strip().split(' ')
            node_id = int(splited[0])
            if node_id not in node2idx:
                dropped_nodes.append(node_id)
                continue

            _features = []
            _labels = []
            for i, x in enumerate(splited[1:]):
                if i in label_indces:
                    _labels.append(int(x))
                else:
                    _features.append(float(x))
            features[node2idx[node_id], :] = torch.tensor(_features, dtype=torch.float)
            labels[node2idx[node_id], :] = torch.tensor(_labels, dtype=torch.int)
    print('The nodes that are not appear in the edges: ', dropped_nodes)

    with open(f'./data/facebook/{ego_id}.egofeat', 'r') as fp:
        line = fp.readline()
        splited = line.strip().split(' ')

        _labels, _features = [], []
        for i, x in enumerate(splited):
            if i in label_indces:
                _labels.append(int(x))
            else:
                _features.append(float(x))
        features[node2idx[ego_id], :] = torch.tensor(_features, dtype=torch.float)
        labels[node2idx[ego_id], :] = torch.tensor(_labels, dtype=torch.int)

    print(f'How many nodes have zero labels: {torch.sum(torch.sum(labels, dim=1) == 0)}')

    # Data = namedtuple('Data', 'x y edge_index')
    # Due to some nodes have zero labels
    # labels = torch.argmax(labels, dim=1)
    labels = torch.argmax(torch.cat((torch.zeros(len(labels), 1), labels), dim=1), dim=1)
    data = TmpData(x=features.float(), y=labels, edge_index=to_undirected(edges))
    return (data, features.size(1), torch.max(labels).item() + 1)

def load_facebook():
    ego_nodes = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980] 
    nodes, edges = set(), set()
    with open('./data/facebook/facebook_combined.txt', 'r') as f:
        for e in f.readlines():
            splited = e.strip().split(' ')
            assert len(splited) == 2, f'Invalid edge: {e}.'
            edges.add(tuple(map(int, splited)))
            nodes.add(int(splited[0]))
            nodes.add(int(splited[1]))
    nodes, edges = list(nodes), list(edges)
    print('*' * 20, 'statistic', '*' * 20)
    print(f'The number of nodes: {len(nodes)}')
    print(f'The number of edges: {len(edges)}')
    print(f'Is the edges undirected: {is_undirected(torch.tensor(edges).t())}')
    print('*' * 50)

    label_name_prefix = 'gender'
    label_names = set()
    feat_names = set()
    for id in ego_nodes:
        with open(f'./data/facebook/{id}.featnames', 'r') as f:
            for line in f.readlines():
                splited = line.strip().split(' ')
                feat_name = ' '.join(splited[1:])
                if feat_name.startswith(label_name_prefix):
                    label_names.add(feat_name)
                else:
                    feat_names.add(feat_name)
    feat_names = list(feat_names)
    feat_name2idx = {name: i for i, name in enumerate(feat_names)}
    label_names = list(label_names)
    label_name2idx = {name: i for i, name in enumerate(label_names)}

    features = torch.zeros(len(nodes), len(feat_names), dtype=torch.int)
    labels = torch.zeros(len(nodes), len(label_names), dtype=torch.int)
    for id in ego_nodes:
        global_feat_indices, local_feat_indices = [], []
        global_label_indices, local_label_indices = [], []
        with open(f'./data/facebook/{id}.featnames', 'r') as f:
            for i, line in enumerate(f.readlines()):
                splited = line.strip().split(' ')
                feat_name = ' '.join(splited[1:])
                if feat_name.startswith(label_name_prefix):
                    global_label_indices.append(label_name2idx[feat_name])
                    local_label_indices.append(i)
                else:
                    global_feat_indices.append(feat_name2idx[feat_name])
                    local_feat_indices.append(i)
        ego_node_feat = []
        ego_label = []

        with open(f'./data/facebook/{id}.egofeat', 'r') as f:
            for i, x in enumerate(f.readline().strip().split(' ')):
                if i in local_feat_indices:
                    ego_node_feat.append(int(x))
                if i in local_label_indices:
                    ego_label.append(int(x))
        assert len(ego_node_feat) == len(local_feat_indices), f'The number of features is not equal to the number of indices for ego id {id}, {len(ego_node_feat)}, {len(local_feat_indices)}.' 
        assert len(ego_label) == len(local_label_indices), f'The number of label is not equal to the number of indices for ego id {id}, {len(ego_label)}, {len(local_label_indices)}.'
        features[int(id), global_feat_indices] = torch.tensor(ego_node_feat, dtype=torch.int)
        labels[int(id), global_label_indices] = torch.tensor(ego_label, dtype=torch.int)

        with open(f'./data/facebook/{id}.feat', 'r') as f:
            for line in f.readlines():
                splited = line.strip().split(' ')
                _id = int(splited[0])
                node_feat = []
                node_label = []
                for i, x in enumerate(splited[1:]):
                    if i in local_feat_indices:
                        node_feat.append(int(x))
                    if i in local_label_indices:
                        node_label.append(int(x))
                assert len(node_feat) == len(local_feat_indices), f'The number of features is not equal to the number of indices for id {_id} of ego node {id}.'
                assert len(node_label) == len(local_label_indices), f'The number of labels not equal to the number of indices for id {_id} of ego node {id}.'
                features[int(_id), global_feat_indices] = torch.tensor(node_feat, dtype=torch.int)
                labels[int(_id), global_label_indices] = torch.tensor(node_label, dtype=torch.int)
    print(f'How many nodes have zero features: {torch.sum(torch.sum(features, dim=1) == 0)}')
    print(f'How many nodes have zero labels: {torch.sum(torch.sum(labels, dim=1) == 0)}')

    filter_out_indices = torch.where(torch.sum(labels, dim=1) != 0)[0].int()
    filtered_nodes = [v for v in range(len(nodes)) if v in filter_out_indices.tolist()]
    fitlered2idx = {v: i for i, v in enumerate(filtered_nodes)}
    edges = [(fitlered2idx[e[0]], fitlered2idx[e[1]]) for e in edges if e[0] in filtered_nodes and e[1] in filtered_nodes]
    features = features[filter_out_indices, :]
    labels = torch.argmax(labels[filter_out_indices, :], dim=1)
    # print(labels[torch.where(torch.sum(labels, dim=1) != 0)[0].int(), :].shape)
    # exit(0)

    Data = namedtuple('Data', 'x y edge_index')

    # Due to some nodes have zero labels
    # labels = torch.argmax(torch.cat((torch.zeros(len(nodes), 1), labels), dim=1), dim=1)
    data = Data(x=features.float(), y=labels, edge_index=to_undirected(torch.tensor(edges).t()))
    return (data, features.size(1), torch.max(labels).item() + 1)


def mocking_graph():
    """
    Create a mocking graph that 
        1. contains 10 nodes (A, B, ..., J), consider A as the target node (test node)
        2. all nodes have the same features with 100 dimensions
        3. 
    """
    # x = torch.rand(2, (1000,)).repeat(10, 1).float()
    x = torch.randint(2, (10, 100)).float()
    y = torch.tensor([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
    # edge_index = torch.tensor([
    #     [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
    #     [1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 7, 8]
    # ])
    edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        [1, 2, 9, 5, 6, 5, 6, 7, 8, 7, 8]
    ])

    Data = namedtuple('Data', 'x y edge_index')
    data = Data(x=x, y=y, edge_index=to_undirected(edge_index))
    return (data, x.size(1), torch.max(y).item() + 1)


class GraphDataset:

    def __init__(self, name, dataset, sample_nodes=None, mock=False, binary=False):
        self.name = name
        self.sample_nodes = sample_nodes
        self.mock = mock

        if isinstance(dataset, tuple):
            self.data = dataset[0]
            self.num_features = dataset[1]
            self.num_classes = dataset[2]
        else:
            self.data = dataset[0]
            self.num_features = dataset.num_features
            self.num_classes = dataset.num_classes

        self.x = self.data.x
        if binary:
            unqiue_labels, counts = torch.unique(self.data.y, return_counts=True)
            print('unqiue_labels:', unqiue_labels, ', counts:', counts)
            largest_label = unqiue_labels[torch.argmax(counts)]
            self.y = torch.where(self.data.y == largest_label, torch.ones_like(self.data.y), torch.zeros_like(self.data.y))
            print('after binary, unqiue_labels:', torch.unique(self.y, return_counts=True))
            self.num_classes = 2
        else:
            self.y = self.data.y
        # features = torch.zeros(len(self.data.x), self.num_features)
        # nn.init.xavier_normal_(features)
        # self.x = features
        # self.y = self.data.y
        # Need to preprocess the edge index
        self.edge_index = self.data.edge_index
        self.edge_index = self.edge_index[:, self.edge_index[0] != self.edge_index[1]] # remove self-loop
        self.edges = self.edge_index.t().tolist()
        if is_undirected(self.edge_index):
            self.edge_index = to_undirected(self.edge_index)

        self.num_nodes = len(self.x)

        if sample_nodes is not None:
            random_nodes = random.sample(range(self.num_nodes), sample_nodes)
            self.x = self.x[random_nodes]
            self.y = self.y[random_nodes]
            self.edge_index = subgraph(torch.tensor(random_nodes, dtype=torch.long), self.edge_index, relabel_nodes=True)[0]
            self.num_nodes = sample_nodes
            print('Sampled nodes:', self.num_nodes)
            print('Sampled edges:', self.edge_index.size(1))
        
        self._split_datasets()
        self.adj_list = self._generate_adj_list()

    def subgraph_by_edges(self, num_edges, force_nodes=None):
        _edges = utils.to_directed(self.edge_index).t().tolist()
        if num_edges < len(_edges):
            _edges = random.sample(_edges, num_edges)
        _nodes = set([v for e in _edges for v in e])
        if force_nodes is not None:
            _nodes = _nodes.union(set(force_nodes))
            # _nodes.extend(force_nodes)
        _nodes = list(_nodes)

        node2idx = {v: idx for idx, v in enumerate(_nodes)}

        self.x = self.x[_nodes]
        self.y = self.y[_nodes]
        _edges = torch.tensor([[node2idx[e[0]], node2idx[e[1]]] for e in _edges], dtype=torch.long)
        self.edge_index = to_undirected(_edges.t())
        # self.edge_index = subgraph(torch.tensor(_nodes, dtype=torch.long), self.edge_index, relabel_nodes=True)[0]
        self.edges = self.edge_index.t().tolist()
        self.num_nodes = len(_nodes)
        self.sample_nodes = self.num_nodes
        self._split_datasets()
        self.adj_list = self._generate_adj_list()

        return node2idx


    def partial_graph(self, size):
        """
        Approahch 1, randomly sample nodes and edges
        """
        # random_nodes = random.sample(range(self.num_nodes), int(size * self.num_nodes))
        # self.x = self.x[random_nodes]
        # self.y = self.y[random_nodes]
        # self.num_nodes = len(random_nodes)
        # self.partial_to_original = {i: v for i, v in enumerate(random_nodes)}
        # self.edge_index = subgraph(torch.tensor(random_nodes, dtype=torch.long), self.edge_index, relabel_nodes=True)[0]
        # self._split_datasets()
        # self.adj_list = self._generate_adj_list()
        """
        Approach 2, randomly fix one node and sample the subgraph in a BFS manner
        """
        anchor = self._find_anchor(list(range(self.num_nodes)))
        subset, self.edge_index = self._bfs_subgraph(anchor, int(size * self.num_nodes), relabel_nodes=True)
        self.x = self.x[subset]
        self.y = self.y[subset]
        self.num_nodes = len(subset)
        self.partial_to_original = {i: v for i, v in enumerate(subset)}
        self._split_datasets()
        self.adj_list = self._generate_adj_list()

    def _find_anchor(self, candidates):
        """
        Find the anchor node that has the most edges
        """
        # print('candidates:', len(candidates))
        # idx2node = {i: v for i, v in enumerate(candidates)}
        # degree = torch.zeros(len(candidates), dtype=torch.int)
        # for i, candidate in enumerate(candidates):
        #     degree[i] = self.degree(candidate)
        # return idx2node[torch.argmax(degree).item()]
        """
        Randomly find an anchor node
        """
        return random.randint(0, self.num_nodes - 1)

    def _bfs_subgraph(self, anchor, size, relabel_nodes=False):
        """
        BFS to sample the subgraph
        """
        col, row = self.edge_index
        node_mask = row.new_empty(self.num_nodes, dtype=torch.bool)
        edge_mask = row.new_empty(self.num_nodes, dtype=torch.bool)

        if isinstance(anchor, int):
            anchor = torch.tensor([anchor], dtype=torch.long)
        elif isinstance(anchor, (list, tuple)):
            anchor = torch.tensor(anchor, dtype=torch.long)
        
        subsets = [anchor]
        sub_size = 1
        while sub_size < size:
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask]) 
            new_subset = torch.cat(subsets).unique()
            if len(new_subset) > sub_size:
                sub_size = len(new_subset)
            else:
                # encounter an isolated graph, randomly pick another anchor
                anchor = self._find_anchor(list(set(range(self.num_nodes)) - set(new_subset.tolist())))
                subsets.append(torch.tensor([anchor], dtype=torch.long))
        
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        subset = subset[:size]

        node_mask.fill_(False)
        node_mask[subset] = True
        edge_mask = node_mask[row] & node_mask[col]
        edge_index = self.edge_index[:, edge_mask]
        if relabel_nodes:
            mapping = row.new_full((self.num_nodes, ), -1)
            mapping[subset] = torch.arange(subset.size(0))
            edge_index = mapping[edge_index]
        
        return subset.tolist(), edge_index



    def _generate_adj_list(self):
        _adj_list = defaultdict(set)
        for u, v in self.edge_index.t().tolist():
            _adj_list[u].add(v)
            _adj_list[v].add(u)
        return _adj_list

    def _split_datasets(self):
        if self.mock:
            nodes_train = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
            nodes_valid = torch.tensor([5, 7])
            nodes_test = torch.tensor([0])
            y_train = self.y[nodes_train]
            y_valid = self.y[nodes_valid]
            y_test = self.y[nodes_test]
        else:
            nodes = torch.arange(self.num_nodes)
            nodes_train, nodes_test, y_train, y_test = train_test_split(
                nodes, self.y, test_size=0.4)
            nodes_train, nodes_valid, y_train, y_valid = train_test_split(
                nodes_train, y_train, test_size=0.1)
            # nodes_train = tmp_idx_train
            # y_train = self.y[nodes_train]
            # nodes_valid = tmp_idx_valid
            # y_valid = self.y[nodes_valid]
            # nodes_test = tmp_idx_test
            # y_test = self.y[nodes_test]

        self.train_set = GNNDataset(nodes_train, y_train)
        self.valid_set = GNNDataset(nodes_valid, y_valid)
        self.test_set = GNNDataset(nodes_test, y_test)

        print('-' * 10, 'Data summary', '-' * 10)
        print('  # of nodes:', self.num_nodes)
        print('  # of edges:', self.edge_index.size(1))
        print('  # of features:', self.num_features)
        print('  binary feature:', torch.unique(self.x).size(0) == 2)
        print('  # of classes:', self.num_classes)
        print('  # of train nodes:', len(nodes_train))
        print('  # of valid nodes:', len(nodes_valid))
        print('  # of test nodes:', len(nodes_test))
        print('-' * 30)

    def print_info(self):
        print('-' * 10, 'Data summary', '-' * 10)
        print('  # of nodes:', self.num_nodes)
        print('  # of edges:', self.edge_index.size(1))
        print('  # of features:', self.num_features)
        print('  # of classes:', self.num_classes)
        print('  # of train nodes:', len(self.train_set.nodes))
        print('  # of valid nodes:', len(self.valid_set.nodes))
        print('  # of test nodes:', len(self.test_set.nodes))
        print('-' * 30)

    def degree(self, v):
        return torch.sum(self.edge_index[0] == v).int().item()
    
    def adjacency_matrix(self):
        if self.sample_nodes:
            return torch.sparse_coo_tensor(self.edge_index, torch.ones(self.edge_index.size(1)), (self.sample_nodes, self.sample_nodes))
        else:
            return torch.sparse_coo_tensor(self.edge_index, torch.ones(self.edge_index.size(1)), (self.num_nodes, self.num_nodes))
            # return to_torch_coo_tensor(self.edge_index)
    
    def update_edge_index_by_adj(self, adj):
        if not adj.is_sparse:
            adj = adj.to_sparse()
        self.edge_index = to_edge_index(adj)[0]
 
    def add_edges(self, edges):
        self.edge_index = torch.cat((self.edge_index, edges), dim=1)
        _edge_index = utils.to_directed(self.edge_index)
        self.edges = _edge_index.t().tolist()
        self.adj_list = self._generate_adj_list()

    def remove_edges(self, edges):
        def _remove_edge(u, v):
            _match = torch.eq(self.edge_index, torch.tensor([u, v]).view(2, -1))
            idx = _match[0] & _match[1]
            self.edge_index = self.edge_index[:, ~idx]
        
        for u, v in edges:
            _remove_edge(u, v)
            _remove_edge(v, u)

        self.adj_list = self._generate_adj_list()

    def has_edge(self, u, v):
        logits = torch.isin(self.edge_index, torch.tensor([u, v]))
        if torch.sum(torch.logical_and(logits[0], logits[1])) > 0:
            return True
        else:
            return False
        
    def neighbors(self, nodes, l):
        def find_neighbors(nodes, j, result):
            _nodes = set()
            if j == 0:
                return
            for v in nodes:
                for u in self.adj_list[v]:
                    result.add(u)
                    _nodes.add(u)
            find_neighbors(_nodes, j - 1, result)

        result = set()
        find_neighbors(nodes, l, result)
        # filter out nodes
        neighbors = np.array(list(result), dtype=np.int32)
        neighbors = neighbors[np.where(~np.in1d(neighbors, np.array(nodes)))]
        return neighbors
    
    def assign_labels(self, nodes, target_label):
        self.y[nodes] = target_label

        train_indices = np.where(np.in1d(self.train_set.nodes, nodes))
        self.train_set.y[train_indices] = target_label
        valid_indices = np.where(np.in1d(self.valid_set.nodes, nodes))
        self.valid_set.y[valid_indices]
        test_indices = np.where(np.in1d(self.test_set.nodes, nodes))
        self.test_set.y[test_indices]

    def detail_info(self, v):
        degree = self.degree(v)
        label = self.y[v]
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(v, 1, self.edge_index)
        label_dist = np.zeros((self.num_classes), dtype=np.int32)
        for u in subset:
            if u == v:
                continue
            label_dist[self.y[u]] += 1
        
        print('-' * 20, '1-hop info', '-' * 20)
        print(' label:', label.item())
        print(' degree:', degree)
        print(' neighbors:', subset)
        print(' neighbor labels:', self.y[subset].tolist())
        print(' label dist:', label_dist.tolist())
        print('=' * 50)
        
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(v, 2, self.edge_index)
        label_dist = np.zeros((self.num_classes), dtype=np.int32)
        for u in subset:
            if u == v:
                continue
            label_dist[self.y[u]] += 1
        
        print('-' * 20, '2-hop info', '-' * 20)
        print(' label:', label.item())
        print(' degree:', degree)
        print(' neighbors:', subset)
        print(' neighbor labels:', self.y[subset].tolist())
        print(' label dist:', label_dist.tolist())
        print('=' * 50)
    
    def label_distribution(self, v, k):
        subset, _, _, _ = k_hop_subgraph(v, k, self.edge_index)
        target_1hop_dist = np.zeros((self.num_classes), dtype=np.int32)
        for u in subset:
            if u == v:
                continue
            target_1hop_dist[self.y[u]] += 1
        return target_1hop_dist
    
    def distance(self, u, v):
        data = Data(edge_index=self.edge_index, num_nodes=self.num_nodes)
        G = to_networkx(data)
        try:
            length = nx.shortest_path_length(G, source=u, target=v)
        except nx.NetworkXNoPath:
            return np.inf
        return length
    
    def _cosine_similarity(self, u, v):
        return torch.dot(self.x[u], self.x[v]) / (torch.norm(self.x[u]) * torch.norm(self.x[v]))
    
    def measure_density(self, G):
        return nx.density(G)
    
    def measure_community_structure(self):
        h = homophily(self.edge_index, self.y)
        # from networkx.algorithms import community
        # # Using Girvan-Newman algorithm to find communities
        # communities_generator = community.girvan_newman(G)
        # top_level_communities = next(communities_generator)
        # communities = sorted(map(sorted, top_level_communities))
        
        # # Print the number of communities and their sizes
        # num_communities = len(communities)
        # community_sizes = [len(c) for c in communities]
        # print(f"Number of Communities: {num_communities}")
        # print(f"Community Sizes: {community_sizes}")
        # return communities
        return h
    
    # Function to plot node degree distribution
    def plot_degree_distribution(self, G):
        from collections import Counter
        degrees = [degree for node, degree in G.degree()]
        degree_count = Counter(degrees)
        deg, cnt = zip(*degree_count.items())
        
        plt.figure(figsize=(8, 6))
        plt.bar(deg, cnt, width=0.8, color='b')
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.yscale('log') # Use log scale for better visibility of distribution
        plt.xscale('log')
        plt.show()

        print(f"Node Degree Distribution: {degree_count}")
        return degree_count
    
    def lp_analysis(self, edges):
        """
        Analyze the label prediction
        1. Calculate the common neighbors
        2. Count the number of paths
        3. Feature proximity
        """
        data = Data(edge_index=self.edge_index, num_nodes=self.num_nodes)
        G = to_networkx(data)

        density = self.measure_density(G)
        print(f'Density: {density}')
        communities = self.measure_community_structure()
        print(f'Communities: {communities}')
        degree_dist = self.plot_degree_distribution(G)
        print(f'Degree Distribution: {degree_dist}')

        # lsp_list, gsp_listm, fh_list = [], [], []
        # # for v in tqdm(range(self.num_nodes)):
        # #     for u in range(v + 1, self.num_nodes):
        # for u, v in edges:
        #     cn = len(set(self.adj_list[v]).intersection(set(self.adj_list[u])))
        #     gsp = len(list(nx.all_simple_paths(G, source=v, target=u, cutoff=2)))
        #     fh = self._cosine_similarity(v, u)
        #     lsp_list.append(cn)
        #     gsp_listm.append(gsp)
        #     fh_list.append(fh)
        #     print(f'v: {v}, u: {u}, cn: {cn}, gsp: {gsp}, fh: {fh}')
        
        # print('Common neighbors:', np.mean(lsp_list), np.std(lsp_list))
        # print('Number of paths:', np.mean(gsp_listm), np.std(gsp_listm))
        # print('Feature proximity:', np.mean(fh_list), np.std(fh_list))



class GNNDataset(Dataset):

    def __init__(self, nodes, y):
        self.nodes = nodes
        self.y = y

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx], self.y[idx]



def load(args, binary=False):
    if args.dataset == 'cora':
        dataset = GraphDataset(args.dataset, Planetoid(root='./data/cora', name='Cora'), sample_nodes=args.subgraph)
    elif args.dataset == 'citeseer':
        dataset = GraphDataset(args.dataset, Planetoid(root='./data/citeseer', name='Citeseer'), binary=binary, sample_nodes=args.subgraph)
    elif args.dataset == 'pubmed':
        dataset = GraphDataset(args.dataset, Planetoid(root='./data/pubmed', name='PubMed'))
    elif args.dataset == 'cs':
        if args.subgraph is None:
            dataset = GraphDataset(args.dataset, Coauthor(root='./data/cs', name='CS'), binary=binary, sample_nodes=10000)
        else:
            with open('./data/cs/subsampled_cs.pkl', 'rb') as fp:
                dataset = pickle.load(fp)
            dataset._split_datasets()
            dataset.edges = dataset.edge_index.t().tolist()
    elif args.dataset == 'polblogs':
        dataset = GraphDataset(args.dataset, PolBlogs(root='./data/polblogs'))

    elif args.dataset == 'facebook':
        dataset = GraphDataset(args.dataset, FacebookPagePage(root='./data/facebook'))
        # dataset = GraphDataset(args.dataset, load_facebook())
        # dataset = GraphDataset(args.dataset, load_single_ego_network())
    elif args.dataset == 'cora-ml':
        dataset = GraphDataset(args.dataset, CitationFull(root='./data/cora_ml', name='cora_ml'))
    elif args.dataset == 'photo':
        dataset = GraphDataset(args.dataset, Amazon(root='./data/photo', name='Photo'))
    elif args.dataset == 'lastfm':
        dataset = GraphDataset(args.dataset, LastFMAsia(root='./data/lastfm'), binary=binary, sample_nodes=args.subgraph)
    elif args.dataset == 'mock':
        dataset = GraphDataset(args.dataset, mocking_graph(), mock=True)
    elif args.dataset == 'mag':
        dataset = GraphDataset(args.dataset, MAG240MDataset(root='./data/mag240m'))
    elif args.dataset == 'ogbn-papers100M':
        dataset = GraphDataset(args.dataset, NodePropPredDataset(name='ogbn-papers100M', root='./data/ogbn-papers100M'))
    else:
        raise NotImplementedError(f'Invalid dataset {args.dataset}.')
    
    return dataset


if __name__ == '__main__':
    x = [19,   9,  29,   5,  17, 100,  11,   3,   2,  31,   9,
          5,  11,   8,   3,   3,  11,   4,   2,   7,  36,  16,  20,
          3,   4,   3,   3,   8,   4,   5,   9,   6,  16,  10,   8,
         35,  12,   9,  11,   3,  16,   4,   9,   4,   3,  10,   2,
          8,   7,   6,  13,   5,  11]
    parser = argument.load_parser()
    parser.add_argument('--node', type=int, required=False)
    parser.add_argument('--nodes', type=int, nargs='+', required=False)
    args = parser.parse_args()

    args.subgraph = 10000
    data = load(args)
    # data.detail_info(args.node)
    # print(data.y[args.nodes].tolist())
    print(torch.unique(data.y, return_counts=True))
    print('!!!', data.x.sum(dim=1).mean())

    from detection import JaccardSimilarity, OutlierDetector
    # detector = JaccardSimilarity(data, [0], [[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]], 'cpu')

    # _data = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
    # G = to_networkx(_data)
    detector = OutlierDetector(data, [0], [[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]])
    # detector.betweenness = nx.betweenness_centrality(G)
    print()

    u = 4875
    # detector.detect_all()
    print(f'finding 0 js with {u}, label {data.y[u]}')
    labels_with_0_js = []
    nodes_with_0_js = []
    for v in range(data.num_nodes):
        if v == u:
            continue
        _v = data.x[v].numpy()
        _u = data.x[u].numpy()
        J = detector._jaccard_similarity(_v, _u)
        if J == 0:
            # print(f'v: {v}, label {data.y[v]}')
            labels_with_0_js.append(data.y[v])
            nodes_with_0_js.append(v)
    print(f'the number of nodes with 0 js: {len(nodes_with_0_js)}')
    print(f'the label distribution:', np.unique(labels_with_0_js, return_counts=True))
    exit(0)

    # for v in range(data.num_nodes):
    #     deg = data.degree(v)
    #     if deg == 28:
    #         print('28', v, deg)
    #     if deg == 18:
    #         print('18', v, deg)
    #     if deg == 8:
    #         print('8', v, deg)
    emmm = [51, 147, 247, 266, 404, 428, 472, 691, 923, 1004, 1202, 1290, 1411, 1471, 1580, 1617, 1724, 1944, 2021, 2226, 2230, 2259, 2282, 2434, 2443, 2444, 2585, 2705, 2774, 2790, 2866, 2919, 3017]
    for v1 in [887, 1429]:
        for v2 in emmm:
            neighbors = data.neighbors([v1, v2, 755], l=1)
            neighbors = [v1, v2, 755] + neighbors.tolist()
            y = [data.degree(v) + 1 for v in neighbors]
            if set(x) == set(y):
                print(v1, v2, neighbors, y)
                break

    # import utils
    # adj = data.adjacency_matrix()
    # adj = utils.sp_add_self_loops(adj)
    # print(adj.is_sparse)
    # utils.normalize(adj)

    # adj = data.adjacency_matrix().to_dense()
    # adj = torch.eye(adj.size(0)) + adj
    # # adj = torch.eye(adj.size(0)) + adj
    # print(adj.is_sparse)
    # utils.normalize(adj)

[19, 9, 29, 5, 17, 12, 100, 11, 31, 2, 5, 9, 11, 8, 3, 11, 4, 7, 36, 16, 20, 3, 4, 3, 5, 8, 4, 5, 6, 16, 10, 8, 2, 35, 12, 11, 9, 3, 16, 4, 9, 4, 3, 5, 10, 8, 7, 6, 13, 5, 11]
[19, 9, 29, 5, 17, 100, 11, 31, 2, 5, 9, 11, 6, 8, 3, 11, 4, 7, 36, 16, 6, 10, 20, 3, 4, 3, 3, 8, 4, 5, 6, 16, 11, 10, 8, 35, 12, 11, 9, 8, 3, 16, 9, 4, 4, 3, 10, 7, 8, 6, 13, 5, 11]