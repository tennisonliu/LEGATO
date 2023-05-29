'''
Constructor of encode/decoder network arrays
'''

from torch import nn

class EncoderArray(nn.Module):
    def __init__(self, encoder_list, freeze=False):
        super(EncoderArray, self).__init__()
        self.encoders = encoder_list
        if freeze:
            for encoder in self.encoders:
                for param in encoder.parameters():
                 param.requires_grad = False

    def forward(self, views):
        '''
        Returns list of view embeddings from list of raw views
            Parameters:
                views (list): list of raw views
            Returns:
                z (list): list of view embeddings, z[i] shape = (b_size, n_feats[i])
        '''
        z = []
        for i, view in enumerate(views):
            feats, _ = self.encoders[i](view)
            z.append(feats)

        return z

class DecoderArray(nn.Module):
    def __init__(self, decoder_list, freeze=False):
        super(DecoderArray, self).__init__()
        self.decoders = decoder_list

        if freeze:
            for decoder in self.decoders:
                for param in decoder.parameters():
                 param.requires_grad = False
    
    def forward(self, embeddings):
        '''
        Returns list of reconstructed views from list of embeddings
            Parameters:
                embeddings (Tensor): view embeddings, shape = (b_size, n_nodes, n_feats)
            Returns:
                x_hat (list): list of view reconstructions
        '''
        num_views = len(self.decoders)
        split_z = [embeddings[:, i, :] for i in range(num_views)]

        x_hat = []
        for i, z in enumerate(split_z):
            x_hat_ = self.decoders[i](z)
            x_hat.append(x_hat_)
        
        return x_hat

class EncoderDecoderArray(nn.Module):
    def __init__(self, encoder_list, decoder_list):
        super(EncoderDecoderArray, self).__init__()
        self.encoders = encoder_list
        self.decoders = decoder_list
        self.name = 'encoder_decoder_array'
    
    def forward(self, views):
        z = []
        for i, view in enumerate(views):
            feats, _ = self.encoders[i](view)
            z.append(feats)

        x_hat = []
        for i, z in enumerate(z):
            x_hat_ = self.decoders[i](z)
            x_hat.append(x_hat_)
        
        return x_hat
