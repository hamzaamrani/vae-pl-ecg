import numpy as np
np.random.seed(0)
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
pl.seed_everything(1234)
import matplotlib.pyplot as plt

import VAE


def create_sine(t, fs, frequency_of_sine):
    return np.sin(2*np.pi*frequency_of_sine*np.arange(0, t, 1./fs))
def create_cosine(t, fs, frequency_of_cosine):
    return np.cos(2*np.pi*frequency_of_cosine*np.arange(0, t, 1./fs))

def prepare_dataset(num_samples = 5000, num_points=1000):
    x_range = np.linspace(0, 2*np.pi, num_points)  # N sample points between 0 and 2π

    t = 10 # s
    fs = 100 # Hz
    N = t*fs

    # Generate dataset
    data = []
    for _ in range(num_samples):
        #A, B = np.random.uniform(0.5, 2, size=2)  # Amplitudes
        #omega, omega_prime = np.random.uniform(1, 5, size=2)  # Frequencies
        #phi, phi_prime = np.random.uniform(0, 2*np.pi, size=2)  # Phases
        #y = A * np.sin(omega * x_range + phi) + B * np.cos(omega_prime * x_range + phi_prime)
        #y = A * np.sin(x_range) + B * np.cos(x_range)

        freq_1, freq_2 = np.random.uniform(0.5, 2, size=2)
        y = create_sine(t, fs, freq_1) + create_sine(t, fs, freq_2)
        data.append(y)
        
        freq_1, freq_2 = np.random.uniform(0.5, 2, size=2)
        y = create_cosine(t, fs, freq_1) + create_cosine(t, fs, freq_2)
        data.append(y)
       

    # Split dataset
    x_train, x_val = train_test_split(np.array(data), test_size=0.2, random_state=42)

    # Convert numpy to PyTorch dataset
    x_train = torch.Tensor(x_train)
    x_val = torch.Tensor(x_val)

    return x_train, x_val

def main():
    # Load dataset
    x_train, x_val = prepare_dataset(num_samples = 10000, num_points=1000)
    train_dataloader = DataLoader(TensorDataset(x_train), batch_size=32) 
    val_dataloader = DataLoader(TensorDataset(x_val), batch_size=32) 

    # Train VAE
    vae = VAE.VAE()
    trainer = pl.Trainer(accelerator="gpu",
                         max_epochs=100)
    trainer.fit(vae, train_dataloader, val_dataloader)


    # TESTING reconstruction
    for i in range(10):
        x = torch.Tensor(x_val[i])
        x_rec = vae(x)

        # Plotting
        plt.figure(figsize=(10, 4))
        plt.plot(x, label="original")
        plt.plot(x_rec, label="rec")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"plots/img_{i}.png")

    a=1

if __name__ == '__main__':
    main()