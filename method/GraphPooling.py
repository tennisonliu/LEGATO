'''
Latent graph pooling module
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .RGCNConv import RGCNConv


class GraphPooling(nn.Module):
    def __init__(self, n_in_feats, n_out_nodes, n_out_feats, n_edge_types,
                 threshold=0.1, gnn_arch='gcn', pool=False, device='cuda'):
        '''
        Constructor for graph pooling module
            Parameters:
                n_in_feats (int): input node feature dimensions
                n_out_nodes (int): size of pooled graph
                n_out_feats (int): pooled node feature dimensions
                n_edge_types (int): number of edge types
                threshold (int): sparsity threshold
                pool (bool): whether to perform pooling or unpooling
        '''
        super(GraphPooling, self).__init__()
        self.n_out_nodes = n_out_nodes
        self.n_out_feats = n_out_feats
        self.device = device
        self.pool = pool

        assert gnn_arch in ['gcn', 'rgcn']
        self.gnn_arch = gnn_arch

        if gnn_arch == 'gcn':
            self.embed = GCNConv(n_in_feats, n_out_feats)
            self.assign_mat = GCNConv(n_in_feats, n_out_nodes)
        else:
            self.embed = RGCNConv(n_in_feats, n_out_feats,
                                  num_relations=n_edge_types+1, root_weight=False)
            # self-loop already included!
            self.assign_mat = RGCNConv(n_in_feats, n_out_nodes,
                                       num_relations=n_edge_types+1, root_weight=False)

        self.act = nn.LeakyReLU()

        self.nodenorm = nn.BatchNorm1d(n_out_feats, affine=False)

        self.threshold = nn.Threshold(threshold, 0)

    def forward(self, processed_input):
        '''
        Returns (un)pooled graph
            Parameters:
                b_z (Tensor): dense feature matrix, shape = (b_size*n_nodes, n_feats)
                b_adj (Tensor): batched adjacency matrix, shape = (b_size, n_nodes, n_nodes)
                b_edge_index (Sparse Tensor): sparse edge index, shape = (2, n_edges)
                b_edge_weights (Sparse Tensor): sparse edge weights, shape = (n_edges)
                b_edge_types (Sparse Tensor): sparse edge type, shape = (n_edges)
            Returns:
                b_z (Tensor): batched feature matrix, shape = (b_size, n_out_nodes, n_out_feats)
                b_adj (Tensor): batched adjacency matrix, shape = (b_size, n_out_nodes, n_out_nodes)
                s_l (Tensor)
        '''
        b_z, b_adj, b_edge_index, b_edge_weights, b_edge_types = processed_input

        (b_size, n_nodes, _) = b_adj.shape

        if self.gnn_arch == 'gcn':
            z_l = self.embed(b_z, b_edge_index, b_edge_weights)
            z_l = self.act(z_l)
            s_l = self.assign_mat(b_z, b_edge_index, b_edge_weights)

        else:
            z_l = self.embed(b_z, b_edge_index, b_edge_types, b_edge_weights)
            z_l = self.act(z_l)
            s_l = self.assign_mat(b_z, b_edge_index, b_edge_types, b_edge_weights)

        z_l = z_l.reshape(b_size, n_nodes, -1)
        s_l = s_l.reshape(b_size, n_nodes, -1)

        if self.pool:
            s_l = F.softmax(s_l, dim=1)
        else:
            s_l = F.softmax(s_l, dim=-1)

        b_z = torch.matmul(s_l.transpose(-1, -2), z_l)
        b_adj = (s_l.transpose(-1, -2)).matmul(b_adj).matmul(s_l)

        b_adj = self.threshold(b_adj)

        assert b_z.shape == torch.Size(
            [b_size, self.n_out_nodes, self.n_out_feats]), f'incorrect shape {b_z.shape}'
        assert b_adj.shape == torch.Size(
            [b_size, self.n_out_nodes, self.n_out_nodes]), f'incorrect shape {b_adj.shape}'

        if self.gnn_arch == 'gcn':
            return (b_z, b_adj), s_l

        else:
            b_z_ = b_z.clone()
            for i in range(b_z.shape[1]):
                b_z_[:, i, :] = self.nodenorm(b_z[:, i, :])

            return (b_z_, b_adj), s_l
