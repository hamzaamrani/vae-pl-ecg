import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, 
                 config):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(config["input_dim"], config["hidden_dim"])
        self.fc2 = nn.Linear(config["hidden_dim"], config["hidden_dim"])

        self.fc_mu = nn.Linear(config["hidden_dim"], config["latent_dim"])
        self.fc_logvar = nn.Linear(config["hidden_dim"], config["latent_dim"])

        self.fc3 = nn.Linear(config["latent_dim"], config["hidden_dim"])
        self.fc4 = nn.Linear(config["hidden_dim"], config["output_dim"])

        #self.dropout = nn.Dropout(config["dropout"])
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        p_x = torch.distributions.Normal(x_hat, scale)
        # measure prob of seeing image under p(x|z)
        log_likelihood = p_x.log_prob(x).sum(-1).mean()
        return log_likelihood

    def kl_divergence(self, z, mu, std):
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # 3. kl loss
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1).mean()
        return kl

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z

    def decode(self, z):
        x_hat = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(x_hat))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    


class VAE_trainer(pl.LightningModule):
    def __init__(self, 
                 config):
        super().__init__()
        self.save_hyperparameters()

        self.beta = config["beta"]
        self.lr = config["lr"]

        self.vae = VAE(config)
        
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def decode(self, x):
        return self.vae.decode(x)
    
    def forward(self, x):
        x_hat, mu, logvar = self.vae(x)
        return x_hat, mu, logvar


    def training_step(self, batch, batch_idx):
        x = batch[0]

        x_hat, mu, logvar = self.vae(x)
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        recon_loss = self.vae.gaussian_likelihood(x_hat, self.vae.log_scale, x)
        kl = self.vae.kl_divergence(z, mu, std)
        loss = -(recon_loss - self.beta*kl)

        self.log("ptl/train_loss", loss, on_step=False, on_epoch=True)
        self.log("ptl/train_loss_kl", kl, on_step=False, on_epoch=True)
        self.log("ptl/train_loss_rec", -recon_loss, on_step=False, on_epoch=True)

        self.logger.experiment.add_scalars('loss', {'train': loss},self.global_step)
        self.logger.experiment.add_scalars('loss_kl', {'train': kl},self.global_step)
        self.logger.experiment.add_scalars('loss_rec', {'train': -recon_loss},self.global_step)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]

        x_hat, mu, logvar = self.vae(x)
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        
        recon_loss = self.vae.gaussian_likelihood(x_hat, self.vae.log_scale, x)
        kl = self.vae.kl_divergence(z, mu, std)
        loss = -(recon_loss - self.beta*kl)

        self.log("ptl/val_loss", loss)
        self.log("ptl/val_loss_kl", kl)
        self.log("ptl/val_loss_rec", -recon_loss)

        self.logger.experiment.add_scalars('loss', {'val': loss},self.global_step)
        self.logger.experiment.add_scalars('loss_kl', {'val': kl},self.global_step)
        self.logger.experiment.add_scalars('loss_rec', {'val': -recon_loss},self.global_step)

        return loss
    

if __name__ == '__main__':
    config = {
        "input_dim": 140,
        "latent_dim": 32,
        "hidden_dim": 64,
        "output_dim": 140,
        "beta": 1,
        "lr": 1e-3,
        "batch_size":32
    }

    model = VAE(config)
    print(model)

    x = torch.ones(32,140)

    with torch.no_grad():
        z = model.encode(x)
        x_hat = model.decode(z)
        x_hat, mu, logvar = model(x)

    BCE, KLD = loss_function(x_hat, x, mu, logvar)