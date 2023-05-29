'''
Utilities to convert data into graph format
'''

import torch
from torch import nn
from torch_geometric.utils import dense_to_sparse

def get_one_hot(view, n_views, b_size):
    '''
    Calculates one-hot encoding for views
    '''
    var_encoding = torch.zeros(b_size, n_views)
    var_encoding[:, [view]] = 1
    return var_encoding


class GraphInputProcessor(nn.Module):
    def __init__(self, n_edge_types, device='cuda'):
        super(GraphInputProcessor, self).__init__()
        self.n_edge_types = n_edge_types
        self.device = device

    def forward(self, z_list, het_encoding=True):
        '''
        Prepares multi-view data for graph processing
            Parameters:
                z_list (list): list of view embeddings
                het_encoding (bool): use of heterogeneous encoding
            Returns:
                b_z (Tensor): batched feature matrix, shape = (b_size, n_nodes, n_feats)
                b_het_z (Tensor): b_z with heterogeneous node encoding
                edge_type (Tensor): edge type encoding, shape = (n_nodes, n_nodes)
        '''
        b_size, n_feats = z_list[0].shape
        n_nodes = len(z_list)

        if het_encoding:
            het_z = []
            for i, z in enumerate(z_list):
                z_encoding = get_one_hot(i, n_nodes, b_size).to(self.device)
                het_z.append(torch.cat([z, z_encoding], dim=1))
            b_het_z = torch.stack(het_z, dim=1)
        else:
            b_het_z = torch.stack(z_list, dim=1)

        b_z = torch.stack(z_list, dim=1)

        n_edge_types = self.n_edge_types
        edge_types = torch.arange(
            1, n_edge_types+1, 1).reshape(n_nodes, n_nodes)

        assert b_z.shape == torch.Size(
            [b_size, n_nodes, n_feats]), f'incorrect shape {b_z.shape}'
        assert b_het_z.shape == torch.Size(
            [b_size, n_nodes, n_feats+n_nodes]), f'incorrect shape {b_het_z.shape}'

        return b_z, b_het_z, edge_types


class GraphEmbeddingProcessor(nn.Module):
    def __init__(self, n_edge_types):
        super(GraphEmbeddingProcessor, self).__init__()
        self.n_edge_types = n_edge_types

    def forward(self, graph_embeddings):
        '''
        Prepares multi-view embeddings for graph processing
            Parameters:
                b_z (Tensor): batched feature matrix, shape = (b_size, n_out_nodes, n_out_feats)
                b_adj (Tensor): batched adjacency matrix, shape = (b_size, n_nodes, n_nodes)
            Returns: 
                b_z (Tensor): dense feature matrix, shape = (b_size*n_nodes, n_feats)
                b_adj (Tensor): batched adjacency matrix, shape = (b_size, n_nodes, n_nodes)
                b_edge_index (Sparse Tensor): sparse edge index, shape = (2, n_edges)
                b_edge_weights (Sparse Tensor): sparse edge weights, shape = (n_edges)
                b_edge_types (Sparse Tensor): sparse edge type, shape = (n_edges)
        '''
        b_z, b_adj = graph_embeddings
        _, _, n_feats = b_z.shape
        b_size, n_nodes, _ = b_adj.shape

        b_z = b_z.reshape(b_size*n_nodes, -1)
        b_edge_index, b_edge_weights = dense_to_sparse(b_adj)

        n_edge_types = self.n_edge_types
        edge_types = torch.arange(
            1, n_edge_types+1, 1).reshape(n_nodes, n_nodes)

        r, c = b_edge_index
        b_edge_types = edge_types[r % n_nodes, c % n_nodes]
        n_edges = b_edge_index.shape[1]

        assert b_z.shape == torch.Size(
            [b_size*n_nodes, n_feats]), f'incorrect shape {b_z.shape}'
        assert b_adj.shape == torch.Size(
            [b_size, n_nodes, n_nodes]), f'incorrect shape {b_adj.shape}'
        assert b_edge_index.shape == torch.Size(
            [2, n_edges]), f'incorrect shape {b_edge_index.shape}'
        assert b_edge_weights.shape == torch.Size(
            [n_edges]), f'incorrect shape {b_edge_weights.shape}'
        assert b_edge_types.shape == torch.Size(
            [n_edges]), f'incorrect shape {b_edge_types.shape}'

        return b_z, b_adj, b_edge_index, b_edge_weights, b_edge_types
