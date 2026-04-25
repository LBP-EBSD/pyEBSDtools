# src/lbp_kikuchi/data/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np

class EBSDDataset(Dataset):
    def __init__(self, X, y, normalize=True):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

        if normalize:
            self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())

        self.X = self.X[:, None, :, :]  # (N,1,H,W)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])