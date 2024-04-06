import numpy as np
np.random.seed(0)
import os
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
pl.seed_everything(1234)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import VAE
from ecg_dataset import ECG5000

def main():
    DIM = 140
    VERSION_DATASET = 4
    BATCH_SIZE = 256

    # Load dataset
    train_dataset = ECG5000('data/ECG5000', phase='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_dataset = ECG5000('data/ECG5000', phase='val')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # VAE model
    # Vanilla VAE
    vae = VAE.VAE(input_dim=DIM,latent_dim=32, hidden_dim=128, output_dim=DIM)
    # Beta-VAE
    #vae = VAE.VAE(input_dim=DIM,latent_dim=64, hidden_dim=512, output_dim=DIM, beta=5)

    # Train model
    trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp",
                         max_epochs=200)
    trainer.fit(vae, train_dataloader, val_dataloader)

    if not os.path.exists("plots"): 
        os.makedirs("plots") 


    # Signal reconstruction
    for j in range(20):
        batch = val_dataset.__getitem__(j)[0].unsqueeze(0)
        batch_pred = vae(batch)

        plt.figure()
        plt.plot(batch[0,:].numpy().flatten(), label="original")
        plt.plot(batch_pred[0,:].numpy().flatten(), label="reconstructed")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"plots/img_{j}.png")

if __name__ == '__main__':
    main()
