import torch
import torch.nn as nn
import torch.nn.functional as F


class StrainHead(nn.Module):
    """
    Regression head: feature embedding → 6 Voigt strain components.

    Voigt order: [ε11, ε22, ε33, ε23, ε13, ε12]

    Architecture: in_dim → hidden_dim → hidden_dim//2 → 6
    Dropout after the first activation reduces overfitting on small datasets.

    Input:  (B, in_dim)
    Output: (B, 6)
    """

    def __init__(self, in_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6),
        )

    def forward(self, x):
        return self.net(x)


class OrientationHead(nn.Module):
    """
    Regression head: feature embedding → unit quaternion (crystal orientation).

    The raw 4-d output is L2-normalized to enforce the unit quaternion
    constraint (q lies on the unit 3-sphere S³). Note that q and -q represent
    the same rotation, so the loss must account for that (see quaternion_geodesic_loss).

    Input:  (B, in_dim)
    Output: (B, 4)  — unit quaternion [q0, q1, q2, q3]
    """

    def __init__(self, in_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x):
        q = self.net(x)
        return F.normalize(q, p=2, dim=-1)


class RelativeStrainHead(nn.Module):
    """
    Phase 2 spatial head: relative feature grid (F_B − F_A) → Δε.

    Uses Conv2D layers instead of MLP so that spatial neighbor structure
    is explicitly modelled (finite-difference / DIC analogy).

    Input:  (B, in_channels, 3, 3)  — relative feature grid
    Output: (B, 6)                  — predicted Δε (Voigt format)
    """

    def __init__(self, in_channels: int = 128, hidden_channels: int = 64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )

    def forward(self, x):
        """
        x:       (B, in_channels, 3, 3)
        returns: (B, 6)
        """
        x = self.conv_layers(x)       # (B, hidden_channels, 3, 3)
        x = x.mean(dim=(-2, -1))      # global average pool → (B, hidden_channels)
        return self.regressor(x)       # (B, 6)


class SpatialStrainHead(nn.Module):
    """
    Stage 2 spatial head: absolute feature grid from one 3×3 neighbourhood → ε(center).

    Same architecture as RelativeStrainHead but operates on absolute (not differenced)
    features. The ConvNet can learn spatial gradients across the 3×3 feature map,
    breaking the ambiguity of predicting absolute strain from a single pattern.

    The skip connection from the center feature (position [1,1]) is concatenated
    with the GAP output before the final regressor, giving the model a direct path
    from the center pattern's embedding to the output.

    Input:  (B, in_channels, 3, 3)  — absolute feature grid
    Output: (B, 6)                  — predicted ε at center (Voigt format)
    """

    def __init__(self, in_channels: int = 128, hidden_channels: int = 64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # regressor takes GAP output + center skip → hidden_channels * 2
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        """
        x:       (B, in_channels, 3, 3)
        returns: (B, 6)
        """
        x = self.conv_layers(x)                    # (B, hidden_channels, 3, 3)
        gap = x.mean(dim=(-2, -1))                 # (B, hidden_channels)
        center = x[:, :, 1, 1]                     # (B, hidden_channels) — center skip
        return self.regressor(torch.cat([gap, center], dim=-1))  # (B, 6)
