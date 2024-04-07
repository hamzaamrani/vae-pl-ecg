import torch
from torch.utils.data import Dataset, DataLoader

import os

import numpy as np
import pandas as pd
from time import sleep
from scipy.io import arff
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import tempfile

def plot_reconstructed_data(vae_best, data):
    test_loader = DataLoader(
        data,
        batch_size=50,
        shuffle=False
    )

    batch = next(iter(test_loader))[0].to(vae_best.device)
    batch_pred = vae_best(batch)[0]

    if not os.path.exists("plots"): 
        os.makedirs("plots") 
    
    for i in range(50):
        plt.figure()
        plt.plot(batch[i,:].cpu().detach().numpy().flatten(), label="original")
        plt.plot(batch_pred[i,:].cpu().detach().numpy().flatten(), label="reconstructed")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"plots/img_{i}.png")

class ECG5000DataModule(pl.LightningDataModule):
    def __init__(self, train, val, test, batch_size=128):
        super().__init__()
        self.data_dir = tempfile.mkdtemp()
        self.batch_size = batch_size

        self.train = train 
        self.val = val #ECG5000("data/ECG5000", phase='val')
        self.test = test #ECG5000("data/ECG5000", phase='test')

    def setup(self, stage=None):
       root = "data/ECG5000"
       #self.train = ECG5000(root, phase='train')
       #self.val = ECG5000(root, phase='val')
       #self.test = ECG5000(root, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=self.batch_size, shuffle=False, num_workers=1)
    

class ECG5000(Dataset):
    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        with open(os.path.join(self.root, 'ECG5000_TRAIN.arff')) as f:
            dataset1, meta1 = arff.loadarff(f)
        with open(os.path.join(self.root, 'ECG5000_TEST.arff')) as f:
            dataset2, meta2 = arff.loadarff(f)
        dataset = pd.concat([pd.DataFrame(dataset1), pd.DataFrame(dataset2)])
        dataset["target"] = pd.to_numeric(dataset["target"])
        if phase == 'train':
            ds = [dataset.loc[dataset['target'] == i].iloc[:-400] for i in [1, 2, 3, 4, 5]]
            dataset = pd.concat(ds, axis=0)
            #dataset = dataset.loc[dataset['target'] == 1].iloc[:-200]
            #dataset2 = dataset.loc[dataset['target'] != 1].iloc[:-200]
            #dataset = pd.concat([dataset1, dataset2], axis=0)
        elif phase == 'val':
            ds = [dataset.loc[dataset['target'] == i].iloc[-400:-200] for i in [1, 2, 3, 4, 5]]
            dataset = pd.concat(ds, axis=0)
            #dataset = dataset.loc[dataset['target'] == 1].iloc[-200:]
        else:
            ds = [dataset.loc[dataset['target'] == i].iloc[-200:] for i in [1, 2, 3, 4, 5]]
            dataset = pd.concat(ds, axis=0)
            #dataset = dataset.loc[dataset['target'] != 1].iloc[:]
        self.dataset = dataset

        super(ECG5000, self).__init__()
    
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        ecg = self.dataset.loc[:, self.dataset.columns != 'target'].iloc[index]
        beat = self.dataset.loc[:, self.dataset.columns == 'target'].iloc[index].values
        ecg = pd.to_numeric(ecg)
        tensor = torch.tensor(ecg, dtype=torch.float32)#.unsqueeze(0)
        
        if self.phase == "train" or self.phase == "val":
            return tensor, beat
        else:
            return tensor, beat
        
        
        
if __name__ == '__main__':
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


    dm = ECG5000DataModule(batch_size=32)


    '''i = iter(train_dataset)
    batch = next(i)
    plt.figure()
    plt.plot(batch.numpy().flatten())
    plt.savefig("plot.png")'''
    # plt.show()