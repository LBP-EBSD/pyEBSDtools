import numpy as np
import torch
from torch.utils.data import Dataset


def compute_norm_stats(X: np.ndarray, method: str) -> dict:
    if method == "minmax":
        return {"min": float(X.min()), "max": float(X.max())}
    elif method == "zscore":
        return {"mean": float(X.mean()), "std": float(X.std())}
    raise ValueError(f"Unknown norm_method: {method!r}")


def apply_norm(X: np.ndarray, stats: dict, method: str) -> np.ndarray:
    if method == "minmax":
        return (X - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)
    elif method == "zscore":
        return (X - stats["mean"]) / (stats["std"] + 1e-8)
    raise ValueError(f"Unknown norm_method: {method!r}")


class EBSDDataset(Dataset):
    """
    Single-pattern dataset. Returns (pattern, targets_dict) per sample.

    Args:
        X:           (N, H, W) raw patterns.
        targets:     Dict of label arrays, e.g. {'strain': (N,6), 'orientation': (N,4)}.
        stats:       Normalisation stats from the training split.
                     If None, computed from X — only pass None for the training split.
        norm_method: 'minmax' or 'zscore'.
    """

    def __init__(
        self,
        X: np.ndarray,
        targets: dict[str, np.ndarray],
        stats: dict | None = None,
        norm_method: str = "minmax",
    ):
        if stats is None:
            stats = compute_norm_stats(X, norm_method)
        self.stats = stats
        self.norm_method = norm_method

        X_norm = apply_norm(X.astype(np.float32), stats, norm_method)
        # Accept both (N, H, W) and (N, 1, H, W) — add channel dim only if needed.
        if X_norm.ndim == 3:
            X_norm = X_norm[:, None]
        self.X = torch.from_numpy(X_norm)  # (N, 1, H, W)
        self.targets = {k: torch.from_numpy(v.astype(np.float32)) for k, v in targets.items()}

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], {k: v[idx] for k, v in self.targets.items()}


class GridPairDataset(Dataset):
    """
    Pairwise 3×3 grid dataset for Phase 2. Returns (grid_a, grid_b, targets_dict).

    Grid patterns are ordered row-major (index 4 = center):
        0 1 2
        3 4 5
        6 7 8

    Args:
        grids_a, grids_b: (N, 9, H, W).
        targets:          Dict of label arrays, typically {'strain': (N,6)} for Δε.
        stats:            Normalisation stats from training split.
                          If None, computed from both grids — only pass None for training.
        norm_method:      'minmax' or 'zscore'.
    """

    def __init__(
        self,
        grids_a: np.ndarray,
        grids_b: np.ndarray,
        targets: dict[str, np.ndarray],
        stats: dict | None = None,
        norm_method: str = "minmax",
    ):
        if stats is None:
            all_pats = np.concatenate(
                [grids_a.reshape(-1, *grids_a.shape[2:]),
                 grids_b.reshape(-1, *grids_b.shape[2:])],
                axis=0,
            )
            stats = compute_norm_stats(all_pats, norm_method)
        self.stats = stats

        ga = apply_norm(grids_a.astype(np.float32), stats, norm_method)
        gb = apply_norm(grids_b.astype(np.float32), stats, norm_method)
        self.grids_a = torch.from_numpy(ga[:, :, None])  # (N, 9, 1, H, W)
        self.grids_b = torch.from_numpy(gb[:, :, None])  # (N, 9, 1, H, W)
        self.targets = {k: torch.from_numpy(v.astype(np.float32)) for k, v in targets.items()}

    def __len__(self) -> int:
        return len(self.grids_a)

    def __getitem__(self, idx: int):
        return self.grids_a[idx], self.grids_b[idx], {k: v[idx] for k, v in self.targets.items()}
