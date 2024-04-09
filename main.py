import numpy as np
np.random.seed(0)
import os
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
pl.seed_everything(1234)

import ray
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

import models
from ecg_dataset import ECG5000, ECG5000DataModule, plot_reconstructed_data

ray.init(num_cpus=1, num_gpus=2)


dataset_train = ECG5000("data/ECG5000", phase='train')
dataset_val = ECG5000("data/ECG5000", phase='val')
dataset_test = ECG5000("data/ECG5000", phase='test')

MODEL_NAME = "vqvae"


def train_func(config):
    dm = ECG5000DataModule(dataset_train, dataset_val, dataset_test, 
                           batch_size=config["batch_size"])
    
    if MODEL_NAME == "vae":
        model = models.VAE(config)
    elif MODEL_NAME == "vqvae":
        model = models.VQVAE(config)
    
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)

def tune_model (search_space, num_epochs = 200, num_samples = 20):
    # num_epochs: The maximum training epochs
    # num_samples: Number of sampls from parameter space
    
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

if __name__ == '__main__':
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
    
    # Hyperparameter tuning
    #results = tune_model(search_space, num_epochs = 1, num_samples = 1)
    results = tune_model(search_space, num_epochs = 500, num_samples = 50)

    # Get best model
    best_result = results.get_best_result(metric="ptl/val_loss", mode="min")
    print(f"best checkpoint: {best_result.checkpoint.path+'/checkpoint.ckpt'}")

    if MODEL_NAME == "vae":
        model_best = models.VAE.load_from_checkpoint(best_result.checkpoint.path+"/checkpoint.ckpt", 
                                            best_result.config['train_loop_config'])
    elif MODEL_NAME == "vqvae":
        model_best = models.VQVAE.load_from_checkpoint(best_result.checkpoint.path+"/checkpoint.ckpt", 
                                            best_result.config['train_loop_config'])
        
    model_best.eval()

    # Plot signal reconstruction
    print("Plotting signals...")
    plot_reconstructed_data(model_best, dataset_test, MODEL_NAME)
    
    print("end!")
    