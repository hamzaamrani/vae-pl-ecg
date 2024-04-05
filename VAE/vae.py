import pytorch_lightning as pl
pl.seed_everything(1234)
from torch import nn
import torch
from torch.nn import functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EncoderRNN, self).__init__()

        self.model = nn.LSTM(1, 128, 2, dropout = 0.2, batch_first=True).to('cuda:0')
                
    def forward(self, x):
        x = torch.unsqueeze(x, 2).to('cuda:0')
        _, (h_end, c_end) = self.model(x)
        h_end = h_end[-1, :, :].to('cuda:0')
        return h_end
    
class DecoderRNN(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(DecoderRNN, self).__init__()

        self.model = nn.LSTM(1, 128, 2).to('cuda:0')
                            
        self.latent_to_hidden = nn.Linear(64, 128).to('cuda:0')
        self.hidden_to_output = nn.Linear(128, 1024).to('cuda:0')

        self.decoder_inputs = torch.zeros(1, 32, 1, requires_grad=True).type(torch.cuda.FloatTensor).to('cuda:0')
        self.c_0 = torch.zeros(2, 32, 128, requires_grad=True).type(torch.cuda.FloatTensor).to('cuda:0')

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)
        
    def forward(self, latent):
        h_state = self.latent_to_hidden(latent).to('cuda:0')

        h_0 = torch.stack([h_state for _ in range(2)]).to('cuda:0')
        decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))


        out = self.hidden_to_output(decoder_output[0,:,:]).to('cuda:0')
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc3 = nn.Linear(hidden_dim//4, hidden_dim//4)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
                
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        #x = self.dropout(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim//4)
        self.fc2 = nn.Linear(hidden_dim//4, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, z):
        x_hat = self.relu(self.fc1(z))
        x_hat = self.dropout(x_hat)
        x_hat = self.relu(self.fc2(x_hat))
        x_hat = self.dropout(x_hat)
        x_hat = self.relu(self.fc3(x_hat))
        #x_hat = self.dropout(x_hat)
        x_hat = self.fc_output(x_hat)
        #x_hat = torch.sigmoid(x_hat)
        
        return x_hat

class VAE(pl.LightningModule):
    def __init__(self, input_dim=1000,latent_dim=64, hidden_dim=512, output_dim=1000, beta=1):
        super().__init__()

        self.save_hyperparameters()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # encoder, decoder
        self.encoder = Encoder(self.input_dim, self.hidden_dim)
        self.decoder = Decoder(self.latent_dim, self.hidden_dim, self.output_dim)

        # distribution parameters
        self.fc_mu = nn.Linear(self.hidden_dim//4, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim//4, self.latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.beta = beta # 1=VAE

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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
    
    def forward(self, x):
        with torch.no_grad():
            z = self.encode(x)
            x_hat = self.decoder(z)
            return x_hat
    
    def encode(self, x):
        with torch.no_grad():
            # encode x to get the mu and variance parameters
            x_encoded = self.encoder(x)
            mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

            # reparameterization trick: sample z from q
            std = torch.exp(log_var / 2)
            eps = torch.randn_like(log_var)  # sampling epsilon        
            z = mu + log_var*eps   
            return z

    def training_step(self, batch, batch_idx):
        x = batch[0]

        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        x_hat = self.decoder(z)

        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        kl = self.kl_divergence(z, mu, std)
        loss = -(recon_loss - self.beta*kl)
        
        self.logger.experiment.add_scalars('loss', 
                                           {'train': loss}, global_step=self.current_epoch)
        self.logger.experiment.add_scalars('kl', 
                                           {'train': kl.mean()}, global_step=self.current_epoch)
        self.logger.experiment.add_scalars('recon_loss', 
                                           {'train': -recon_loss.mean()}, global_step=self.current_epoch)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]

        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        x_hat = self.decoder(z)

        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        kl = self.kl_divergence(z, mu, std)
        loss = -(recon_loss - self.beta*kl)

        self.logger.experiment.add_scalars('loss', 
                                           {'val': loss}, global_step=self.current_epoch)
        self.logger.experiment.add_scalars('kl', 
                                           {'val': kl.mean()}, global_step=self.current_epoch)
        self.logger.experiment.add_scalars('recon_loss', 
                                           {'val': -recon_loss.mean()}, global_step=self.current_epoch)

        return loss


if __name__ == '__main__':
    model = VAE()

    x = torch.ones(32,64)
    z = model.encode(x)