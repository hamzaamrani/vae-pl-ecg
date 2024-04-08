import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
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
        vq_loss = torch.mean((z_q.detach()-z)**2)
        commitmnet_loss = self.beta * torch.mean((z_q - z.detach()) ** 2)

        loss = vq_loss + commitmnet_loss

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(config)
        self.pre_quantization = nn.Linear(config['hidden_dim2'], config['embedding_dim'])
        
        # codebooks - pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(config)
        
        # decode the discrete latent representation
        self.decoder = Decoder(config)

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        return embedding_loss, x_hat, perplexity
    

class VQVAE_trainer(pl.LightningModule):
    def __init__(self, 
                 config):
        super().__init__()
        self.save_hyperparameters()

        self.lr = config["lr"]
        self.model = VQVAE(config)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def forward(self, x):
        embedding_loss, x_hat, perplexity = self.model(x)
        return embedding_loss, x_hat, perplexity

    def training_step(self, batch, batch_idx):
        x = batch[0]

        embedding_loss, x_hat, perplexity = self.model(x)
        #recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        recon_loss = F.mse_loss(x, x_hat)
        loss = recon_loss + embedding_loss

        self.log("ptl/train_loss", loss, on_step=False, on_epoch=True)
        self.log("ptl/train_loss_embedding", embedding_loss, on_step=False, on_epoch=True)
        self.log("ptl/train_loss_rec", recon_loss, on_step=False, on_epoch=True)

        self.logger.experiment.add_scalars('loss', {'train': loss},self.global_step)
        self.logger.experiment.add_scalars('loss_loss_embedding', {'train': embedding_loss},self.global_step)
        self.logger.experiment.add_scalars('loss_rec', {'train': recon_loss},self.global_step)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]

        embedding_loss, x_hat, perplexity = self.model(x)
        #recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        recon_loss = F.mse_loss(x, x_hat)
        loss = recon_loss + embedding_loss

        self.log("ptl/val_loss", loss)
        self.log("ptl/val_loss_embedding", embedding_loss)
        self.log("ptl/val_loss_rec", recon_loss)

        self.logger.experiment.add_scalars('loss', {'val': loss},self.global_step)
        self.logger.experiment.add_scalars('loss_loss_embedding', {'val': embedding_loss},self.global_step)
        self.logger.experiment.add_scalars('loss_rec', {'val': recon_loss},self.global_step)

        return loss

if __name__ == "__main__":
    config = {
        "input_dim":140,
        "hidden_dim1":128,
        "hidden_dim2":64,

        "n_embeddings":512,
        "embedding_dim":64,
        "beta":.25,

        "lr": 1e-3,
        "batch_size":32,
        "dropout":0
    }
    # beta: ranging from 0.1 to 2.0

    

    # random data
    x = np.random.random_sample((32, 140))
    x = torch.tensor(x).float()

    model = VQVAE(config)
    print(model)

    embedding_loss, x_hat, perplexity = model(x)

    
    a=1
