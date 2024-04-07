import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.
    """

    def __init__(self, config):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim1']),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config['hidden_dim1'], config['hidden_dim1']),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config['hidden_dim1'], config['hidden_dim2']),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)
    
class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.
    """

    def __init__(self, config):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(config['embedding_dim'], config['hidden_dim2']),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config['hidden_dim2'], config['hidden_dim1']),
            nn.GELU(),
            nn.Linear(config['hidden_dim1'], config['input_dim'])
        )

    def forward(self, x):
        return self.layers(x)

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, config):
        super(VectorQuantizer, self).__init__()
        
        self.n_e = config['n_embeddings']
        self.e_dim = config['embedding_dim']
        self.beta = config['beta']

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return loss, z_q, perplexity, min_encodings, min_encoding_indices