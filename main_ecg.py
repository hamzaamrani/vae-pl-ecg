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

import VAE
from ecg_dataset import ECG5000, ECG5000DataModule


dataset_train = ECG5000("data/ECG5000", phase='train')
dataset_val = ECG5000("data/ECG5000", phase='val')
dataset_test = ECG5000("data/ECG5000", phase='test')

def train_func(config):
    dm = ECG5000DataModule(dataset_train, dataset_val, dataset_test, 
                           batch_size=config["batch_size"])
    
    model = VAE.VAE(config)
    
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

if __name__ == '__main__':
    # Configuring the search space
    search_space = {
        "input_dim": 140,
        "latent_dim": tune.choice([64, 32]),
        "hidden_dim": tune.choice([64, 128]),
        "output_dim": 140,
        "beta": 1,
        #"dropout": tune.uniform(0, 0.9),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size":tune.choice([32, 64, 128, 256]),
    }

    import ray
    ray.init(num_cpus=1, num_gpus=2)
    
    # The maximum training epochs
    num_epochs = 200
    # Number of sampls from parameter space
    num_samples = 20
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
    
    results.get_best_result(metric="ptl/val_loss", mode="min")
    


    '''
    train_dataset = ECG5000('data/ECG5000', phase='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True
    )
    val_dataset = ECG5000('data/ECG5000', phase='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )

    config = {
        "input_dim": 140,
        "latent_dim": 32,
        "hidden_dim": 64,
        "output_dim": 140,
        "beta": 1,
        "lr": 1e-3,
        "batch_size":32
    }

    vae = VAE.VAE(config)

    # Train model
    trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp",
                         max_epochs=10)
    trainer.fit(vae, train_loader, val_loader)
    '''

    '''if not os.path.exists("plots"): 
        os.makedirs("plots") 
    
    # Signal reconstruction
    for j in range(20):
        batch = dm.test.__getitem__(j)[0].unsqueeze(0)
        batch_pred = vae(batch)

        plt.figure()
        plt.plot(batch[0,:].numpy().flatten(), label="original")
        plt.plot(batch_pred[0,:].numpy().flatten(), label="reconstructed")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"plots/img_{j}.png")'''