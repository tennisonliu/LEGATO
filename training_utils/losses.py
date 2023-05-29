'''
Unsupervised training objective
'''

import torch
from torch import nn

class RegLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1):
        super(RegLoss, self).__init__()
        self.recon_criterion = nn.MSELoss(reduction='mean')
        self.cos_criterion = nn.CosineSimilarity(dim=-1, eps=1e-3)
        self.beta = beta
        self.alpha = alpha

    def cos_loss(self, embeddings):
        '''
        orthogonality loss
        '''
        b_size, n_nodes, _ = embeddings.shape

        cos_loss = self.cos_criterion(
            embeddings[:, None, :, :], embeddings[:, :, None, :])
        assert cos_loss.shape == torch.Size(
            [b_size, n_nodes, n_nodes]), f'incorrect shape {cos_loss.shape}'

        cos_loss = cos_loss.abs() - torch.eye(n_nodes).to(embeddings.device).repeat(b_size, 1, 1)
        cos_loss = cos_loss.sum()/(b_size*n_nodes*(n_nodes-1))
        return cos_loss

    def graph_loss(self, input_graph):
        '''
        sparsity loss
        '''
        if len(input_graph.shape) == 3:
            b_size, n_nodes, _ = input_graph.shape
        else:
            _, b_size, n_nodes, _ = input_graph.shape

        graph_loss = torch.norm(
            input_graph-torch.eye(n_nodes, n_nodes).to(input_graph.device), p=1)/(b_size)
        return graph_loss

    def recon_graph_loss(self, input_graph, recon_graph):
        '''
        grpah reconstruction loss
        '''
        if len(input_graph.shape) > 3:
            input_graph = input_graph.mean(dim=0).squeeze()

        assert input_graph.shape == recon_graph.shape, f'oops...'
        b_size, _, _ = input_graph.shape

        graph_loss = torch.norm(input_graph-recon_graph, p=2)
        av_graph_loss = graph_loss/b_size

        return av_graph_loss

    def forward(self, views, views_recon, latent_embeddings, input_graph, recon_graph):
        recon_loss = 0
        for i, view in enumerate(views):
            recon_loss = recon_loss + \
                self.recon_criterion(view, views_recon[i])

        recon_loss = recon_loss/len(views)
        recon_graph_loss = self.recon_graph_loss(input_graph, recon_graph)

        cos_loss = self.cos_loss(latent_embeddings)
        graph_loss = self.graph_loss(input_graph)

        loss = recon_loss + recon_graph_loss + \
            self.alpha*cos_loss + self.beta*graph_loss
        return loss, recon_loss, recon_graph_loss, cos_loss, graph_loss
