import os
import random

import numpy as np

np.random.seed(0)

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

pl.seed_everything(1234)


import ray
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (RayDDPStrategy, RayLightningEnvironment,
                                 RayTrainReportCallback, prepare_trainer)
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

import models
from ecg_dataset import ECG5000, ECG5000DataModule, plot_reconstructed_data

# Initialize Ray with specified resources
ray.init(num_cpus=1, num_gpus=2)

# Load ECG5000 dataset for training, validation, and testing
dataset_train = ECG5000("data/ECG5000", phase='train')
dataset_val = ECG5000("data/ECG5000", phase='val')
dataset_test = ECG5000("data/ECG5000", phase='test')

# Define the model name to be used
MODEL_NAME = "pae"

def train_func(config):
    # Initialize the data module with the given configuration
    dm = ECG5000DataModule(dataset_train, dataset_val, dataset_test, 
                           batch_size=config["batch_size"])
    
    # Select the model based on the MODEL_NAME
    if MODEL_NAME == "vae":
        model = models.VAE(config)
    elif MODEL_NAME == "vqvae":
        model = models.VQVAE(config)
    elif MODEL_NAME == "pae":
        model = models.PAE(config)
    
    # Initialize the PyTorch Lightning trainer with Ray integration
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False
    )
    # Prepare the trainer for Ray
    trainer = prepare_trainer(trainer)
    # Fit the model using the trainer and data module
    trainer.fit(model, datamodule=dm)

def tune_model(search_space, num_epochs=200, num_samples=20):
    # num_epochs: The maximum training epochs
    # num_samples: Number of samples from parameter space
    
    # This scheduler decides at each iteration which trials are likely to perform badly, and stops these trials.
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    # Training with GPUs
    scaling_config = ScalingConfig(
        num_workers=2, 
        use_gpu=True, 
        resources_per_worker={"GPU": 1}
    )
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_loss",
            checkpoint_score_order="min",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    def tune_asha(num_samples=10):
        scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric="ptl/val_loss",
                mode="min",
                num_samples=num_samples,
                scheduler=scheduler,
            )
        )
        return tuner.fit()
    
    results = tune_asha(num_samples=num_samples)

    return results

def main():
    if MODEL_NAME == "vae":
        search_space = {
            "input_dim": 140,
            "latent_dim": tune.choice([32, 64]),
            "hidden_dim": tune.choice([64, 128]),
            "hidden_dim2": tune.choice([32, 64]),
            "output_dim": 140,
            "beta": tune.choice([1, 5, 10, 50, 100]),
            "dropout": tune.uniform(0, 0.9),
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size":tune.choice([32, 64, 128, 256]),
        }
    elif MODEL_NAME == "vqvae":
        search_space = {
            "input_dim": 140,
            "hidden_dim1": tune.choice([64, 128]),
            "hidden_dim2": tune.choice([32, 64]),
            "n_embeddings":tune.choice([256, 512]),
            "embedding_dim":tune.choice([32, 64]),
            "beta": tune.uniform(0.1, 2.),
            "dropout": tune.uniform(0, 0.9),
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size":tune.choice([32, 64, 128, 256]),
        }
    elif MODEL_NAME == "pae":
        search_space = {
            "input_channels": 1,
            "embedding_channels": tune.choice([2, 3, 4, 5]), #desired number of latent phase channels (usually between 2-10)
            "time_range": 141,
            "window": 1,
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size":tune.choice([32, 64, 128, 256]),
            }
    
    # Hyperparameter tuning
    results = tune_model(search_space, num_epochs = 500, num_samples = 10)

    # Get best model
    best_result = results.get_best_result(metric="ptl/val_loss", mode="min")
    print(f"best checkpoint: {best_result.checkpoint.path+'/checkpoint.ckpt'}")

    if MODEL_NAME == "vae":
        model_best = models.VAE.load_from_checkpoint(best_result.checkpoint.path+"/checkpoint.ckpt", 
                                            best_result.config['train_loop_config'])
    elif MODEL_NAME == "vqvae":
        model_best = models.VQVAE.load_from_checkpoint(best_result.checkpoint.path+"/checkpoint.ckpt", 
                                            best_result.config['train_loop_config'])
    elif MODEL_NAME == "pae":
        model_best = models.PAE.load_from_checkpoint(best_result.checkpoint.path+"/checkpoint.ckpt", 
                                            best_result.config['train_loop_config'])
    
    model_best.eval()

    # Plot signal reconstruction
    print("Plotting signals...")
    plot_reconstructed_data(model_best, dataset_test, MODEL_NAME)
    
    print("end!")

if __name__ == '__main__':
    main()
    