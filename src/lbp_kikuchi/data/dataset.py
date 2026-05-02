import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Spatial dataset builders
# ---------------------------------------------------------------------------


def build_grid_samples(
    X: np.ndarray,
    y_strain: np.ndarray,
    grid_rows: int,
    grid_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 3×3 grid samples for Stage 2 training.

    Extracts a 3×3 neighbourhood of patterns centred on each *interior* scan
    point (i.e. every point that has a full ring of 8 neighbours). Boundary
    points are discarded.

    The flat index ``i`` is assumed to map to scan position
    ``(i // grid_cols, i % grid_cols)`` — i.e. row-major / C order, which is
    the natural ordering when EMsoft writes patterns in scan order.

    Args:
        X:                (N, H, W) flat pattern array.
                          Must satisfy N == grid_rows * grid_cols.
        y_strain:         (N, 6) absolute Voigt strain labels.
        grid_rows:        Number of scan rows.
        grid_cols:        Number of scan columns.

    Returns:
        grids:   (M, 9, H, W)  M = (grid_rows-2) * (grid_cols-2) interior points
        labels:  (M, 6)        ε at the centre of each grid (index 4)
    """
    N, H, W = X.shape
    if N != grid_rows * grid_cols:
        raise ValueError(
            f"X has {N} patterns but grid_rows*grid_cols = {grid_rows * grid_cols}."
        )

    X_2d = X.reshape(grid_rows, grid_cols, H, W)
    y_2d = y_strain.reshape(grid_rows, grid_cols, 6)

    grids, labels = [], []
    for r in range(1, grid_rows - 1):
        for c in range(1, grid_cols - 1):
            patch = X_2d[r - 1 : r + 2, c - 1 : c + 2]   # (3, 3, H, W)
            grids.append(patch.reshape(9, H, W))
            labels.append(y_2d[r, c])

    return np.stack(grids), np.stack(labels)


def build_pair_samples(
    X: np.ndarray,
    y_strain: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    directions: tuple[str, ...] = ("horizontal", "vertical"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build pairs of 3×3 grids for Stage 3 (Δε) training.

    Each pair consists of two *adjacent interior* scan points A and B.
    The label is Δε = ε_B − ε_A (Voigt, 6 components).

    Both A and B must be interior points (full 3×3 neighbourhood available),
    so the usable range shrinks by one additional step in the pair direction.

    Args:
        X:          (N, H, W) flat pattern array.
        y_strain:   (N, 6) absolute Voigt strain labels.
        grid_rows:  Number of scan rows.
        grid_cols:  Number of scan columns.
        directions: Which adjacency directions to include.
                    'horizontal' → A=(r,c), B=(r,c+1)
                    'vertical'   → A=(r,c), B=(r+1,c)

    Returns:
        grids_a:      (M, 9, H, W)
        grids_b:      (M, 9, H, W)
        delta_strain: (M, 6)  Δε = ε_B − ε_A
    """
    N, H, W = X.shape
    if N != grid_rows * grid_cols:
        raise ValueError(
            f"X has {N} patterns but grid_rows*grid_cols = {grid_rows * grid_cols}."
        )

    X_2d = X.reshape(grid_rows, grid_cols, H, W)
    y_2d = y_strain.reshape(grid_rows, grid_cols, 6)

    ga_list: list[np.ndarray] = []
    gb_list: list[np.ndarray] = []
    dy_list: list[np.ndarray] = []

    if "horizontal" in directions:
        # A=(r,c), B=(r,c+1); both need full 3×3 neighbourhoods.
        # A: c ∈ [1, cols-2],  B: c+1 ∈ [1, cols-2] → c ∈ [1, cols-3]
        for r in range(1, grid_rows - 1):
            for c in range(1, grid_cols - 2):
                ga_list.append(X_2d[r - 1 : r + 2, c - 1 : c + 2].reshape(9, H, W))
                gb_list.append(X_2d[r - 1 : r + 2, c : c + 3].reshape(9, H, W))
                dy_list.append(y_2d[r, c + 1] - y_2d[r, c])

    if "vertical" in directions:
        # A=(r,c), B=(r+1,c); both need full 3×3 neighbourhoods.
        # A: r ∈ [1, rows-2],  B: r+1 ∈ [1, rows-2] → r ∈ [1, rows-3]
        for r in range(1, grid_rows - 2):
            for c in range(1, grid_cols - 1):
                ga_list.append(X_2d[r - 1 : r + 2, c - 1 : c + 2].reshape(9, H, W))
                gb_list.append(X_2d[r : r + 3, c - 1 : c + 2].reshape(9, H, W))
                dy_list.append(y_2d[r + 1, c] - y_2d[r, c])

    if not ga_list:
        raise ValueError(
            f"No pairs found. Check grid dimensions ({grid_rows}×{grid_cols}) "
            f"and directions {directions}."
        )

    return np.stack(ga_list), np.stack(gb_list), np.stack(dy_list)


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


class GridDataset(Dataset):
    """
    Single 3×3 grid dataset for Stage 2. Returns (grid, targets_dict) per sample.

    The interface mirrors EBSDDataset — batch items are (grid, targets_dict) —
    so this dataset is compatible with the existing train_one_epoch / evaluate
    engine functions without modification.

    Grid patterns are ordered row-major (index 4 = center):
        0 1 2
        3 4 5
        6 7 8

    Args:
        grids:       (N, 9, H, W) — one 3×3 grid per interior scan point.
                     Build with build_grid_samples().
        targets:     Dict of label arrays, e.g. {'strain': (N, 6)} for ε at center.
        stats:       Normalisation stats (computed from training split patterns).
                     If None, computed from grids — only pass None for the training split.
        norm_method: 'minmax' or 'zscore'.
    """

    def __init__(
        self,
        grids: np.ndarray,
        targets: dict[str, np.ndarray],
        stats: dict | None = None,
        norm_method: str = "minmax",
    ):
        N, n_pats, H, W = grids.shape   # n_pats = 9
        if stats is None:
            flat = grids.reshape(-1, H, W)
            stats = compute_norm_stats(flat, norm_method)
        self.stats = stats
        self.norm_method = norm_method

        g = apply_norm(grids.astype(np.float32), stats, norm_method)
        self.grids = torch.from_numpy(g[:, :, None])    # (N, 9, 1, H, W)
        self.targets = {k: torch.from_numpy(v.astype(np.float32)) for k, v in targets.items()}

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, idx: int):
        return self.grids[idx], {k: v[idx] for k, v in self.targets.items()}


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
