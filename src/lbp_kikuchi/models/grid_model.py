import torch
import torch.nn as nn
from lbp_kikuchi.models.encoder import Encoder
from lbp_kikuchi.models.heads import SpatialStrainHead


class GridModel(nn.Module):
    """
    Stage 2 model: single 3×3 local pattern grid → absolute strain ε(center).

    Adds spatial context over Stage 1 by processing the full 3×3 neighbourhood
    jointly. Neighbouring patterns break the ambiguity of predicting absolute
    strain from a single pattern because the model can now learn spatial gradients
    — differences between neighbours are a proxy for strain variation.

    Input:
        grid: (B, 9, 1, H, W)  — 9 patterns in 3×3 neighbourhood

    Output:
        dict with 'strain' → (B, 6)  — predicted ε at center (Voigt format)

    Grid ordering (row-major, index 4 = center):
        0 1 2
        3 4 5
        6 7 8

    Architecture:
        Shared Encoder (same weights for all 9 patterns)
        → feature map: (B, feature_dim, 3, 3)
        → SpatialStrainHead (Conv2D → GAP + center-skip → FC) → (B, 6)

    The center skip in SpatialStrainHead gives the model a direct path from the
    center pattern's embedding to the output, while the convolutional path learns
    spatial gradients across the neighbourhood.
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(out_dim=feature_dim)
        self.head = SpatialStrainHead(in_channels=feature_dim)

    def _encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Encode all 9 patterns in a 3×3 grid using the shared encoder.

        grid:    (B, 9, 1, H, W)
        returns: (B, feature_dim, 3, 3)
        """
        B, N, C, H, W = grid.shape  # N = 9
        flat = grid.view(B * N, C, H, W)        # (B*9, 1, H, W)
        features = self.encoder(flat)            # (B*9, feature_dim)
        features = features.view(B, N, -1)       # (B, 9, feature_dim)
        features = features.permute(0, 2, 1)     # (B, feature_dim, 9)
        return features.view(B, -1, 3, 3)        # (B, feature_dim, 3, 3)

    def forward(self, grid: torch.Tensor) -> dict:
        """
        grid:    (B, 9, 1, H, W)
        returns: dict with 'strain' → (B, 6)
        """
        feat = self._encode_grid(grid)           # (B, C, 3, 3)
        return {"strain": self.head(feat)}       # (B, 6)

    def get_grid_features(self, grid: torch.Tensor) -> torch.Tensor:
        """Return (B, feature_dim, 3, 3) feature map for analysis / visualisation."""
        return self._encode_grid(grid)
