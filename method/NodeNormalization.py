'''
Node normalization scheme
'''

from torch import nn
from torch.nn import BatchNorm1d

class NodeNormalization(nn.Module):
    def __init__(self, n_in_feats):
        '''
        Constructor for node-wise normalization module
            Parameters: 
                n_in_feats (int): number of input features
        '''
        super(NodeNormalization, self).__init__()
        self.nodenorm = BatchNorm1d(n_in_feats)
    
    def forward(self, z_list):
        '''
        Returns node-wise normalized feature matrix
            Parameters:
                z_list (list): list of view embeddings
            Returns:
                norm_z_list (list): list of normalized view embeddings
        '''
        norm_z_list = []
        for _, view in enumerate(z_list):
            norm_z_list.append(self.nodenorm(view))
        return norm_z_list