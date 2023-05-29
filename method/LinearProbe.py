'''
Linear probe for downstream learning
'''

from torch import nn

class LinearProbe(nn.Module):
    def __init__(self, encoder, n_in_feats, n_targets, freeze_encoder=True, aggr='mean'):
        super(LinearProbe, self).__init__()

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        print(f'finetuning with frozen encoder={freeze_encoder}...')
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.aggr = aggr
        if aggr in ['mean', 'max', 'sum']:
            n_in_feats = n_in_feats
        else:
            n_in_feats = n_in_feats * self.encoder.n_nodes
        self.linear_probe = nn.Linear(n_in_feats, n_targets)

    def forward(self, x):
        self.encoder.eval()
        _, latent_graph, _, _ = self.encoder(x)

        z = latent_graph[0]
        b_size, _, n_feats = z.shape
        if self.aggr == 'mean':
            z = z.mean(dim=1)
        elif self.aggr == 'max':
            z = z.max(dim=1)
        elif self.aggr == 'sum':
            z = z.sum(dim=1)
        else:
            z = z.reshape(b_size, -1)

        y_hat = self.linear_probe(z)

        return y_hat
