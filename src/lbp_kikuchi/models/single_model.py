import torch.nn as nn
from lbp_kikuchi.models.encoder import Encoder
from lbp_kikuchi.models.heads import StrainHead, OrientationHead


class SinglePatternModel(nn.Module):
    """
    Phase 1 model: single Kikuchi pattern → strain (+ optionally orientation).

    Input:  (B, 1, H, W)
    Output: dict with keys:
        'strain':      (B, 6)   always present — Voigt strain components
        'orientation': (B, 4)   present only if predict_orientation=True

    Architecture:
        Encoder (ResNet18 → feature_dim)
        ├── StrainHead      → (B, 6)
        └── OrientationHead → (B, 4)   [optional]

    The encoder is designed to be reused as-is in Phase 2 (PairModel),
    where the same weights encode all 9 patterns in each 3×3 grid.

    Args:
        feature_dim:          Size of the shared feature embedding.
        predict_orientation:  Whether to attach and train the orientation head.
    """

    def __init__(self, feature_dim: int = 128, predict_orientation: bool = False,
                 img_size: int = 224):
        super().__init__()
        self.predict_orientation = predict_orientation

        self.encoder = Encoder(out_dim=feature_dim, img_size=img_size)
        self.strain_head = StrainHead(in_dim=feature_dim)

        if predict_orientation:
            self.orientation_head = OrientationHead(in_dim=feature_dim)

    def forward(self, x) -> dict:
        """
        x:       (B, 1, H, W)
        returns: dict of output tensors
        """
        features = self.encoder(x)
        out = {"strain": self.strain_head(features)}

        if self.predict_orientation:
            out["orientation"] = self.orientation_head(features)

        return out

    def get_features(self, x):
        """Return raw feature embeddings without running heads."""
        return self.encoder(x)
