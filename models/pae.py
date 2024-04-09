import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y

class PAE(nn.Module):
    def __init__(self, config):
        super(PAE, self).__init__()

        config = {
            "input_channels": 1,
            "embedding_channels": 5,
            "time_range": 140+1,
            "window": 1,

            "lr": 1e-3,
            "batch_size":32,
            "dropout":0
        }
        input_channels = config["input_channels"]
        embedding_channels = config["embedding_channels"]
        time_range = config["time_range"]
        window = config["window"]

        self.input_channels = config["input_channels"]
        self.embedding_channels = config["embedding_channels"]
        self.time_range = config["time_range"]
        self.window = config["window"]

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(time_range)[1:] * time_range / self.window, requires_grad=False) #Remove DC frequency

        intermediate_channels = int(input_channels/3)
        if intermediate_channels == 0:
            intermediate_channels = 1
        
        self.conv1 = nn.Conv1d(input_channels, intermediate_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.norm1 = LN_v2(time_range)
        self.conv2 = nn.Conv1d(intermediate_channels, embedding_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.fc = torch.nn.ModuleList()
        for i in range(embedding_channels):
            self.fc.append(nn.Linear(time_range, 2))

        self.deconv1 = nn.Conv1d(embedding_channels, intermediate_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.denorm1 = LN_v2(time_range)
        self.deconv2 = nn.Conv1d(intermediate_channels, input_channels, time_range, stride=1, padding=int((time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

    #Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        #Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        #Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        #Offset
        offset = rfft.real[:,:,0] / self.time_range #DC component

        return freq, amp, offset

    def forward(self, x):
        y = x

        #Signal Embedding
        #y = y.reshape(y.shape[0], self.input_channels, self.time_range)

        y = self.conv1(y)
        y = self.norm1(y)
        y = F.elu(y)

        y = self.conv2(y)

        latent = y #Save latent for returning

        #Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        #Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi

        #Parameters    
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b] #Save parameters for returning

        #Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b

        signal = y #Save signal for returning

        #Signal Reconstruction
        y = self.deconv1(y)
        y = self.denorm1(y)
        y = F.elu(y)

        y = self.deconv2(y)

        #y = y.reshape(y.shape[0], self.input_channels*self.time_range)

        return y, latent, signal, params

class PAE_trainer(pl.LightningModule):
    def __init__(self, 
                 config):
        super().__init__()
        self.save_hyperparameters()

        self.lr = config["lr"]

        self.pae = PAE(config)
        
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


    def forward(self, x):
        y, latent, signal, params = self.pae(x)
        return y, latent, signal, params


    def training_step(self, batch, batch_idx):
        x = batch[0]

        x = x.unsqueeze(1)
        x = F.pad(x, (0,1), "constant", 0)

        yPred, latent, signal, params = self.pae(x)
        loss = F.mse_loss(yPred, x)
        
        self.log("ptl/train_loss", loss, on_step=False, on_epoch=True)
        self.logger.experiment.add_scalars('loss', {'train': loss},self.global_step)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]

        x = x.unsqueeze(1)
        x = F.pad(x, (0,1), "constant", 0)

        yPred, latent, signal, params = self.pae(x)
        loss = F.mse_loss(yPred, x)

        self.log("ptl/val_loss", loss)
        self.logger.experiment.add_scalars('loss', {'val': loss},self.global_step)

        return loss


if __name__ == '__main__':

    config = {
        "input_channels": 1,
        "embedding_channels": 5, #desired number of latent phase channels (usually between 2-10)
        "time_range": 140+1,
        "window": 1,

        "lr": 1e-4,
        "batch_size":32,
        "dropout":0
    }

    model = PAE(config)
    print(model)

    x = torch.ones(32,140)

    y, latent, signal, params = model(x)