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

import models


def prepare_dataset(num_samples = 5000, num_points=1000, version = 3):
    x_values = np.linspace(0, 2*np.pi, num_points)  # N sample points between 0 and 2π

    # Generate dataset
    data = []
    for _ in range(num_samples):        

        # 1st experiment: baseline + random noise - good result
        if version == 1:
            sin_values = np.sin(2 * np.pi * np.linspace(0, 4, 1024)) #4s 256Hz
            cos_values = np.cos(2 * np.pi * np.linspace(0, 4, 1024)) #4s 256Hz
            combined_values = sin_values + cos_values
            noise = np.random.normal(0, 0.1, combined_values.shape)
            y = combined_values + noise

        # 2nd experiment: baseline + random noise + modulation - good result
        elif version == 2:
            time_base = np.linspace(0, 4, 1024) #4s 256Hz
            sin_values = np.sin(2 * np.pi * time_base)
            cos_values = np.cos(2 * np.pi * time_base) 
            combined_values = sin_values + cos_values
            modulation_frequency = 0.5  # Hz
            modulation_signal = np.sin(2 * np.pi * modulation_frequency * time_base)
            combined_values = (1 + modulation_signal) * combined_values
            noise = np.random.normal(0, 0.1, combined_values.shape)
            y = combined_values + noise#'''

        #3rd experiment: baseline + random noise (more variable) + random modulation
        elif version == 3: 
            time_base = np.linspace(0, 4, 1024) #4s 256Hz
            sin_values = np.sin(2 * np.pi * time_base)
            cos_values = np.cos(2 * np.pi * time_base)
            combined_values = sin_values + cos_values
            # Random amplitude modulation
            modulation_frequency = np.random.normal(0.5, 0.1)  # Hz, drawn from a normal distribution
            modulation_signal = np.sin(2 * np.pi * modulation_frequency * time_base)
            combined_values = (1 + modulation_signal) * combined_values
            # Complex noise addition
            noise_level = np.linspace(0.05, 0.15, len(time_base))  # Increasing noise level over time
            noise = np.random.normal(0, noise_level, combined_values.shape)
            y = combined_values + noise#'''

        #4th experiment: baseline + random noise (more variability) + random modulation (more variability) + random phase shifts
        else:
            time_base = np.linspace(0, 4, 1024) 
            # Base frequencies for sine and cosine components
            base_frequencies = np.array([1])  # For example, 1Hz and 2Hz
            phase_shifts = np.random.uniform(0, 2*np.pi, size=base_frequencies.shape)  # Random phase shifts
            
            combined_values = np.zeros_like(time_base)
            for base_freq, phase_shift in zip(base_frequencies, phase_shifts):
                sin_component = np.sin(2 * np.pi * base_freq * time_base + phase_shift)
                cos_component = np.cos(2 * np.pi * base_freq * time_base + phase_shift)
                combined_values += sin_component + cos_component
            
            # Random amplitude modulation for each base frequency
            modulation_frequencies = np.random.normal(0.5, 0.2, size=base_frequencies.shape)  # More variability
            for mod_freq in modulation_frequencies:
                modulation_signal = np.sin(2 * np.pi * mod_freq * time_base)
                combined_values *= (1 + modulation_signal)
            
            # Complex noise addition
            noise_level = np.linspace(0.05, 0.15, len(time_base))  # Increasing noise level over time
            noise = np.random.normal(0, noise_level, combined_values.shape)
            combined_values += noise
            y = combined_values#'''

        data.append(y)
       

    # Split dataset
    x_train, x_val = train_test_split(np.array(data), test_size=0.2, random_state=42)

    # Normalization
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    # Convert numpy to PyTorch dataset
    x_train = torch.Tensor(x_train)
    x_val = torch.Tensor(x_val)

    return x_train, x_val, scaler

def main():
    DIM = 1024
    VERSION_DATASET = 4

    # Load dataset
    x_train, x_val, scaler = prepare_dataset(num_samples = 20000, num_points=DIM, version = VERSION_DATASET)
    train_dataloader = DataLoader(TensorDataset(x_train), batch_size=32, num_workers=3) 
    val_dataloader = DataLoader(TensorDataset(x_val), batch_size=32, num_workers=3) 

    # VAE model
    # Vanilla VAE
    #vae = VAE.VAE(input_dim=DIM,latent_dim=64, hidden_dim=512, output_dim=DIM)
    # Beta-VAE
    vae = models.VAE(input_dim=DIM,latent_dim=64, hidden_dim=512, output_dim=DIM, beta=5)

    # Train model
    trainer = pl.Trainer(accelerator="gpu", devices=1,#devices=2, strategy="ddp",
                         max_epochs=100)
    trainer.fit(vae, train_dataloader, val_dataloader)

    if not os.path.exists("plots"): 
        os.makedirs("plots") 

    # Signal reconstruction
    for i in range(20):
        x = torch.Tensor(x_val[i])
        x_rec = vae(x)

        x = scaler.inverse_transform([x.numpy()])[0]
        x_rec = scaler.inverse_transform([x_rec.numpy()])[0]

        # Plotting
        plt.figure(figsize=(10, 4))
        plt.plot(x, label="original")
        plt.plot(x_rec, label="rec")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"plots/img_{i}.png")

if __name__ == '__main__':
    main()
