import numpy as np
np.random.seed(0)
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
pl.seed_everything(1234)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import VAE


def create_sine(t, fs, frequency_of_sine):
    return np.sin(2*np.pi*frequency_of_sine*np.arange(0, t, 1./fs))
def create_cosine(t, fs, frequency_of_cosine):
    return np.cos(2*np.pi*frequency_of_cosine*np.arange(0, t, 1./fs))

def prepare_dataset(num_samples = 5000, num_points=1000):
    x_values = np.linspace(0, 2*np.pi, num_points)  # N sample points between 0 and 2π

    '''t = 10 # s
    fs = 100 # Hz
    N = t*fs'''

    # Generate dataset
    data = []
    for _ in range(num_samples):
        #A, B = np.random.uniform(0.5, 2, size=2)  # Amplitudes
        #omega, omega_prime = np.random.uniform(1, 5, size=2)  # Frequencies
        #phi, phi_prime = np.random.uniform(0, 2*np.pi, size=2)  # Phases
        #y = A * np.sin(omega * x_range + phi) + B * np.cos(omega_prime * x_range + phi_prime)
        #y = A * np.sin(x_range) + B * np.cos(x_range)

        '''freq_1, freq_2 = np.random.uniform(0.5, 2, size=2)
        y = create_sine(t, fs, freq_1) + create_sine(t, fs, 0.5)
        data.append(y)
        
        freq_1, freq_2 = np.random.uniform(0.5, 2, size=2)'''

        '''A = np.random.normal(1., 0.5)
        B = np.random.normal(1., 0.5)

        y = A*create_sine(t, fs, 0.5) + B*create_cosine(t, fs, 0.5)'''

        '''# 1st experiment: simple -> TOP
        sin_values = np.sin(x_values)
        cos_values = np.cos(x_values)
        combined_values = sin_values + cos_values
        noise = np.random.normal(0, 0.1, combined_values.shape)
        y = combined_values + noise'''

        # 1st experiment: baseline + random noise - good result
        '''sin_values = np.sin(2 * np.pi * np.linspace(0, 4, 1024)) #4s 256Hz
        cos_values = np.cos(2 * np.pi * np.linspace(0, 4, 1024)) #4s 256Hz
        combined_values = sin_values + cos_values
        noise = np.random.normal(0, 0.1, combined_values.shape)
        y = combined_values + noise'''

        # 2nd experiment: baseline + random noise + modulation - good result
        '''time_base = np.linspace(0, 4, 1024) #4s 256Hz
        sin_values = np.sin(2 * np.pi * time_base)
        cos_values = np.cos(2 * np.pi * time_base) 
        combined_values = sin_values + cos_values
        modulation_frequency = 0.5  # Hz
        modulation_signal = np.sin(2 * np.pi * modulation_frequency * time_base)
        combined_values = (1 + modulation_signal) * combined_values
        noise = np.random.normal(0, 0.1, combined_values.shape)
        y = combined_values + noise'''

        #3rd experiment: baseline + random noise (more variable) + random modulation
        '''time_base = np.linspace(0, 4, 1024) #4s 256Hz
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
        time_base = np.linspace(0, 4, 1024) 
        # Base frequencies for sine and cosine components
        base_frequencies = np.array([1, 2])  # For example, 1Hz and 2Hz
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
    # Load dataset
    x_train, x_val, scaler = prepare_dataset(num_samples = 20000, num_points=DIM)
    train_dataloader = DataLoader(TensorDataset(x_train), batch_size=64, num_workers=3) 
    val_dataloader = DataLoader(TensorDataset(x_val), batch_size=64, num_workers=3) 

    # TESTING reconstruction
    for i in range(10):
        x = torch.Tensor(x_val[i])
        x = scaler.inverse_transform([x.numpy()])[0]

        # Plotting
        plt.figure(figsize=(10, 4))
        plt.plot(x, label="original")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"plots/img_{i}.png")

    # Train VAE
    vae = VAE.VAE(input_dim=DIM,latent_dim=64, hidden_dim=512, output_dim=DIM)
    trainer = pl.Trainer(accelerator="gpu",devices=2, strategy="ddp",
                         max_epochs=100)
    trainer.fit(vae, train_dataloader, val_dataloader)


    # TESTING reconstruction
    for i in range(10):
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

    a=1

if __name__ == '__main__':
    main()