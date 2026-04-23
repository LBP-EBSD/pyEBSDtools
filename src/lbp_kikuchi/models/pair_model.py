import torch.nn as nn
from lbp_kikuchi.models.encoder import Encoder

class PairModel(nn.Module):
    def __init__(self, feat_dim=64, out_dim=1):
        super().__init__()
        self.encoder = Encoder(feat_dim)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        return self.head(f2 - f1)