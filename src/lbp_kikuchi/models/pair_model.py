import torch
import torch.nn as nn
from lbp_kikuchi.models.encoder import Encoder
from lbp_kikuchi.models.heads import RelativeStrainHead


class PairModel(nn.Module):
    """
    Phase 2 model: two 3×3 local pattern grids → relative strain Δε.

    Mirrors the HR-EBSD approach:
        strain ≈ small relative distortion between patterns
        → compute features of each grid, subtract, predict Δε.

    Input:
        grid_a: (B, 9, 1, H, W)  — 9 patterns in 3×3 neighborhood at point A
        grid_b: (B, 9, 1, H, W)  — 9 patterns in 3×3 neighborhood at point B
    Output:
        delta_strain: (B, 6)     — predicted Δε = ε_B − ε_A (Voigt format)

    Grid ordering (row-major):
        0 1 2
        3 4 5
        6 7 8

    Architecture:
        Shared Encoder (same weights for all 18 patterns)
        → grid A features: (B, C, 3, 3)
        → grid B features: (B, C, 3, 3)
        → relative features: F_B − F_A  (B, C, 3, 3)
        → RelativeStrainHead (Conv2D → GAP → FC) → (B, 6)

    Why subtraction?
        Cancels shared orientation/intensity bias, isolates deformation signal.
        Analogous to DIC (digital image correlation) in full-field strain measurement.
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(out_dim=feature_dim)
        self.head = RelativeStrainHead(in_channels=feature_dim)

    def _encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Encode all 9 patterns in a 3×3 grid using the shared encoder.

        grid:    (B, 9, 1, H, W)
        returns: (B, feature_dim, 3, 3)
        """
        B, N, C, H, W = grid.shape      # N = 9

        flat = grid.view(B * N, C, H, W)         # (B*9, 1, H, W)
        features = self.encoder(flat)             # (B*9, feature_dim)
        features = features.view(B, N, -1)        # (B, 9, feature_dim)
        features = features.permute(0, 2, 1)      # (B, feature_dim, 9)
        return features.view(B, -1, 3, 3)         # (B, feature_dim, 3, 3)

    def forward(self, grid_a: torch.Tensor, grid_b: torch.Tensor) -> dict:
        """
        grid_a, grid_b: (B, 9, 1, H, W)
        returns: dict with key 'strain' → (B, 6)
        """
        feat_a = self._encode_grid(grid_a)    # (B, C, 3, 3)
        feat_b = self._encode_grid(grid_b)    # (B, C, 3, 3)

        rel = feat_b - feat_a                 # (B, C, 3, 3)
        return {"strain": self.head(rel)}     # (B, 6)

    def get_grid_features(self, grid: torch.Tensor) -> torch.Tensor:
        """Return (B, feature_dim, 3, 3) grid features for analysis."""
        return self._encode_grid(grid)
