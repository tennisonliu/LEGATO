'''
Multiview input-graph learner
'''

import torch
from torch import nn
from torch_geometric.utils import dense_to_sparse

class LearnerHead(nn.Module):
    def __init__(self, n_in_feats, n_out_feats, threshold=0.1):
        super(LearnerHead, self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(n_in_feats, n_out_feats)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.threshold = nn.Threshold(threshold, 0)

        self.act = nn.LeakyReLU()
        self.cos_criterion = nn.CosineSimilarity(dim=-1, eps=1e-3)

    def forward(self, b_het_z):
        h = torch.matmul(b_het_z, self.W)
        h = self.act(h)
        b_adj = self.cos_criterion(h[:, None, :, :], h[:, :, None, :]).abs()
        b_adj = self.threshold(b_adj)
        return b_adj


class GraphLearner(nn.Module):
    def __init__(self, n_in_feats, n_out_feats=10, threshold=0.1, n_heads=3, device='cuda'):
        '''
        Constructor for graph learner module
            Parameters:
                n_heads (int): number of self-attention heads
                threshold (float): sparsity threshold
        '''
        super(GraphLearner, self).__init__()
        self.heads = nn.ModuleList(LearnerHead(n_in_feats, n_out_feats, threshold).to(device) for _ in range(n_heads))
        self.count = 0

    def forward(self, processed_input):
        '''
        Returns weighted adjacency matrix between views using self-attention mechanism
            Parameters:
                b_z (Tensor): batched feature matrix, shape = (b_size, n_nodes, n_feats)
                b_het_z (Tensor): b_z with heterogeneous node encoding
                edge_type (Tensor): edge type encoding, shape = (n_nodes, n_nodes)
            Returns:
                b_z (Tensor): dense feature matrix, shape = (b_size*n_nodes, n_feats)
                b_adj (Tensor): batched adjacency matrix, shape = (b_size, n_nodes, n_nodes)
                b_edge_index (Sparse Tensor): sparse edge index, shape = (2, n_edges)
                b_edge_weights (Sparse Tensor): sparse edge weights, shape = (n_edges)
                b_edge_types (Sparse Tensor): sparse edge type, shape = (n_edges)
        '''
        b_z, b_het_z, edge_types = processed_input
        
        b_size, n_nodes, n_feats = b_z.shape

        b_adj_ = torch.stack([head(b_het_z) for head in self.heads])
        b_adj = b_adj_.mean(dim=0)

        b_edge_index, b_edge_weights = dense_to_sparse(b_adj)

        r, c = b_edge_index
        b_edge_types = edge_types[r%n_nodes, c%n_nodes]

        b_z = b_z.reshape(b_size*n_nodes, n_feats)

        n_edges = b_edge_index.shape[1]

        assert b_edge_index.shape == torch.Size([2, n_edges]), f'incorrect shape {b_edge_index.shape}'
        assert b_edge_types.shape == torch.Size([n_edges]), f'incorrect shape {b_edge_types.shape}'
        assert b_edge_weights.shape == torch.Size([n_edges]), f'incorrect shape {b_edge_weights.shape}'
        assert b_z.shape == torch.Size([b_size*n_nodes, n_feats]), f'incorrect shape {b_z.shape}'
        assert b_adj.shape == torch.Size([b_size, n_nodes, n_nodes]), f'incorrect shape {b_adj.shape}'

        # NOTE: b_adj and b_edge_weights contains the same gradients but in different formats for graph pooling
        return (b_z, b_adj, b_edge_index, b_edge_weights, b_edge_types), b_adj_
