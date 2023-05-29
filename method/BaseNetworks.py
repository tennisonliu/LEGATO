'''
Base encoder/decoder networks
'''

import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, layer_size, num_layers=1):
        super(Encoder, self).__init__()

        network = nn.ModuleList()
        current_dim = input_dim
        for i in range(num_layers):
            network.append(nn.Linear(current_dim, layer_size))
            network.append(nn.ReLU(inplace=False))
            current_dim = layer_size
        # mu and logvar so output_dim*2
        network.append(nn.Linear(layer_size, int(output_dim*2)))

        self.network = nn.Sequential(*network)

    def forward(self, x):
        out = self.network(x)
        assert len(out.shape) == 2, 'oops'

        a, b = out.split(int(out.shape[1]/2), dim=1)
        return a, b


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, layer_size, num_layers=1):
        super(Decoder, self).__init__()

        network = nn.ModuleList()
        current_dim = input_dim
        for i in range(num_layers):
            network.append(nn.Linear(current_dim, layer_size))
            network.append(nn.ReLU(inplace=False))
            current_dim = layer_size
        network.append(nn.Linear(layer_size, int(output_dim)))

        self.network = nn.Sequential(*network)

    def forward(self, x):
        out = self.network(x)
        assert len(out.shape) == 2, 'oops'

        return out