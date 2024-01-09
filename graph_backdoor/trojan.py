from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import copy
import sys
import os
# from utils.datareader import DataReader

# from utils.mask import recover_mask
# from trojan.prop import forwarding


def forwarding(args, bkd_adj, model, bkd_data, criterion, device, for_whom):
    # gdata = GraphData(bkd_dr, gids)
    # loader = DataLoader(gdata,
    #                     batch_size=args.batch_size,
    #                     shuffle=False,
    #                     collate_fn=collate_batch)
    loader = DataLoader(bkd_data.train_set, batch_size=args.gta_batch_size, shuffle=False)
    edge_index = torch.cat(torch.where(bkd_adj == 1)).view(2, -1).to(device)
    _data = copy.deepcopy(bkd_data)
    _data.edge_index = edge_index
    # if for_whom == 'feat':
    #     model.update_embedding(bkd_feat)

    # if not next(model.parameters()).is_cuda:
    # model = model.to(device)
    # model.eval()
    # all_loss, n_samples = 0.0, 0.0
    # for nodes, _ in loader:
    #     labels = torch.tensor(data.labels[nodes], device=device)
    #     nodes = nodes.to(device)

    #     output = model(nodes, edge_index)
    #     if len(output.shape) == 1:
    #         output = output.unsqueeze(0)

    #     loss = criterion(output, labels)  # only calculate once
    #     all_loss = torch.add(torch.mul(loss, len(output)), all_loss)  # cannot be loss.item()
    #     n_samples += len(output)

    # all_loss = torch.div(all_loss, n_samples)
    return model.loss(_data, loader, criterion, device)


class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input > thrd, torch.tensor(1.0, device=device, requires_grad=True),
                          torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None


class GraphTrojanNet(nn.Module):
    def __init__(self, sq_dim, layernum=1, dropout=0.05):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum - 1):
            layers.append(nn.Linear(sq_dim, sq_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(sq_dim, sq_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, input, mask, thrd,
                device=torch.device('cpu'),
                activation='relu',
                for_whom='topo',
                binaryfeat=False):
        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """
        GW = GradWhere.apply

        bkdmat = self.layers(input)
        if activation == 'relu':
            bkdmat = F.relu(bkdmat)
        elif activation == 'sigmoid':
            bkdmat = torch.sigmoid(bkdmat)    # nn.Functional.sigmoid is deprecated

        if for_whom == 'topo':  # not consider direct yet
            bkdmat = torch.div(torch.add(bkdmat, bkdmat.transpose(0, 1)), 2.0)
        if for_whom == 'topo' or (for_whom == 'feat' and binaryfeat):
            bkdmat = GW(bkdmat, thrd, device)
        bkdmat = torch.mul(bkdmat, mask)

        return bkdmat


def train_gtn(args, model, toponet: GraphTrojanNet, featnet: GraphTrojanNet,
              topo_input, feat_input, topomask, featmask, data, adj, bkd_adj, device):
    """
    All matrix/array like inputs should already in torch.tensor format.
    All tensor parameters or models should initially stay in CPU when
    feeding into this function.

    About inputs of this function:
    - pset/nset: gids in trainset
    - init_dr: init datareader, keep unmodified inside of each resampling
    - bkd_dr: store temp adaptive adj/features, get by  init_dr + GTN(inputs)
    """

    # i = torch.tensor(data.edges)
    # v = torch.ones(len(data.edges))
    # adj = torch.sparse_coo_tensor(list(zip(*i)), v, (data.num_nodes, data.num_nodes), device=device).to_dense()
    # bkd_adj = copy.deepcopy(adj)

    features = data.x.to(device)
    optimizer_topo = optim.Adam(toponet.parameters(),
                                lr=args.gtn_lr,
                                weight_decay=5e-4)
    if args.feat_perb:
        bkd_features = copy.deepcopy(features)
        optimizer_feat = optim.Adam(featnet.parameters(),
                                    lr=args.gtn_lr,
                                    weight_decay=5e-4)

    #----------- training topo generator -----------#
    toponet = toponet.to(device)
    # model = model.to(device)
    topo_thrd = torch.tensor(args.topo_thrd, device=device)
    criterion = nn.CrossEntropyLoss()

    toponet.train()
    for e in tqdm(range(args.gtn_epochs), desc="training topology generator"):
        optimizer_topo.zero_grad()
        # generate new adj_list by dr.data['adj_list']
        rst_bkdA = toponet(topo_input, topomask, topo_thrd, device, args.topo_activation, 'topo')
        bkd_adj = torch.add(rst_bkdA, adj)   # only current position in cuda

        loss = forwarding(args, bkd_adj, model, data, criterion, device, 'topo')
        loss.backward()
        optimizer_topo.step()

    toponet.eval()
    topo_gen = toponet.cpu()
    del toponet
    del topo_thrd
    # torch.cuda.empty_cache()

    if args.feat_perb:
        featnet = featnet.to(device)
        feat_thrd = torch.tensor(args.feat_thrd, device=device)
        criterion = nn.CrossEntropyLoss()

        featnet.train()
        for epoch in tqdm(range(args.gtn_epochs), desc="training feature generator"):
            optimizer_feat.zero_grad()
            # generate new features by dr.data['features']
            # for gid in pset:
            # SendtoCUDA(gid, [init_Xs, Xinputs, featmasks])  # only send the used graph items to cuda
            rst_bkdX = featnet(
                feat_input, featmask, feat_thrd, device, args.feat_activation, 'feat')
            # rst_bkdX = recover_mask(nodenums[gid], featmasks[gid], 'feat')
            # bkd_dr.data['features'][gid] = torch.add(rst_bkdX, init_Xs[gid])
            bkd_features = torch.add(rst_bkdX, features)   # only current position in cuda
            # SendtoCPU(gid, [init_Xs, Xinputs, featmasks])

            # generate DataLoader
            loss = forwarding(
                args, bkd_adj, bkd_features, model, data, criterion, device, for_whom='feat')
            loss.backward()
            optimizer_feat.step()
            torch.cuda.empty_cache()

        featnet.eval()
        featnet.cpu()
        del feat_thrd
        torch.cuda.empty_cache()

    return topo_gen, featnet

# ----------------------------------------------------------------


def SendtoCUDA(gid, items):
    """
    - items: a list of dict / full-graphs list, 
             used as item[gid] in items
    - gid: int
    """
    cuda = torch.device('cuda')
    for item in items:
        item[gid] = torch.as_tensor(item[gid], dtype=torch.float32).to(cuda)


def SendtoCPU(gid, items):
    """
    Used after SendtoCUDA, target object must be torch.tensor and already in cuda.

    - items: a list of dict / full-graphs list, 
             used as item[gid] in items
    - gid: int
    """

    cpu = torch.device('cpu')
    for item in items:
        item[gid] = item[gid].to(cpu)
