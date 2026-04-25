# src/lbp_kikuchi/models/heads.py

import torch.nn as nn

class StrainHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.net(x)