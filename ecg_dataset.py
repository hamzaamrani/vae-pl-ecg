import torch
from torch.utils.data import Dataset, DataLoader

import os

import numpy as np
import pandas as pd
from time import sleep
from scipy.io import arff
import matplotlib.pyplot as plt


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
            dataset = dataset.loc[dataset['target'] == 1].iloc[:-200]
        elif phase == 'val':
            dataset = dataset.loc[dataset['target'] == 1].iloc[-200:]
        else:
            dataset = dataset.loc[dataset['target'] != 1]
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

    from ecg_dataset import FakeECG, ECG5000
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




    i = iter(train_dataset)
    batch = next(i)
    plt.figure()
    plt.plot(batch.numpy().flatten())
    plt.savefig("plot.png")
    # plt.show()