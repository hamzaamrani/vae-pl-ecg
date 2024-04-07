import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


from VAE.vqvae_utils import Encoder
from VAE.vqvae_utils import VectorQuantizer
from VAE.vqvae_utils import Decoder



class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(config)
        self.pre_quantization = nn.Linear(config['hidden_dim2'], config['embedding_dim'])
        
        # pass continuous latent vector through discretization bottleneck
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
