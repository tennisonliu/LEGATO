'''
Main architecture of LEGATO
'''

import math
from torch import nn

from .EncoderDecoderArray import EncoderArray, DecoderArray
from .GraphLearner import GraphLearner
from .GraphPooling import GraphPooling
from .NodeNormalization import NodeNormalization
from .GraphDataUtils import GraphInputProcessor, GraphEmbeddingProcessor


class LEGATO(nn.Module):
    def __init__(self, n_views, n_in_feats,
                 encoder_list, decoder_list,
                 gnn_arch='gcn',
                 pool_ratio=1, sparse_threshold=0.1,
                 device='cuda'):
        super(LEGATO, self).__init__()

        self.name = 'legato'

        self.encoder_array = EncoderArray(encoder_list, freeze=False)
        self.decoder_array = DecoderArray(decoder_list, freeze=False)

        self.nodenorm = NodeNormalization(n_in_feats=n_in_feats)

        n_edge_types = int(n_views**2)
        self.graph_input_processor = GraphInputProcessor(
            n_edge_types=n_edge_types, device=device)

        self.graph_learner = GraphLearner(
            n_in_feats=n_in_feats+n_views,
            n_out_feats=100,
            threshold=sparse_threshold,
            n_heads=1,
            device=device
        )

        n_nodes = max(math.ceil(n_views*pool_ratio), 1)
        self.n_nodes = n_nodes

        n_feats = int(n_in_feats / pool_ratio)
        self.pool = GraphPooling(
            n_in_feats, n_nodes,
            n_feats, n_edge_types,
            gnn_arch=gnn_arch, pool=True
        )

        n_edge_types = int(n_nodes**2)
        self.graph_embedding_processor = GraphEmbeddingProcessor(
            n_edge_types=n_edge_types)

        self.unpool = GraphPooling(
            n_feats, n_views,
            n_in_feats, n_edge_types,
            gnn_arch=gnn_arch, pool=False
        )

        self.latent_dims = n_feats

    def forward(self, views):
        z_list = self.encoder_array(views)
        norm_z_list = self.nodenorm(z_list)
        in_graph = self.graph_input_processor(norm_z_list)
        learned_graph, input_graph = self.graph_learner(in_graph)
        latent_graph, assignment_mat = self.pool(learned_graph)
        embedding_input = self.graph_embedding_processor(latent_graph)
        recon_graph, unassignment_mat = self.unpool(embedding_input)
        x_hat = self.decoder_array(recon_graph[0])

        return x_hat, latent_graph, input_graph, (assignment_mat, unassignment_mat, recon_graph[1])
