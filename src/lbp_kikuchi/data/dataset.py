import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Spatial dataset builders
# ---------------------------------------------------------------------------


def _squeeze_channel(X: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Accept both (N, H, W) and (N, 1, H, W) pattern arrays.
    Returns (X_3d, H, W) with shape (N, H, W).

    X_patterns.npy is written channel-first (N, 1, H, W) by convert.py.
    The build functions work on (N, H, W) then add the channel dim back
    inside the Dataset __init__ via the [:, :, None] indexing.
    """
    if X.ndim == 4:
        if X.shape[1] != 1:
            raise ValueError(
                f"Expected X with 1 channel, got shape {X.shape}. "
                "Pass (N, H, W) or (N, 1, H, W)."
            )
        X = X[:, 0]   # (N, 1, H, W) → (N, H, W)
    if X.ndim != 3:
        raise ValueError(f"Expected 3-D or 4-D X, got shape {X.shape}.")
    _, H, W = X.shape
    return X, H, W


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
        X:                (N, H, W) or (N, 1, H, W) flat pattern array.
                          Must satisfy N == grid_rows * grid_cols.
        y_strain:         (N, 6) absolute Voigt strain labels.
        grid_rows:        Number of scan rows.
        grid_cols:        Number of scan columns.

    Returns:
        grids:   (M, 9, H, W)  M = (grid_rows-2) * (grid_cols-2) interior points
        labels:  (M, 6)        ε at the centre of each grid (index 4)
    """
    X, H, W = _squeeze_channel(X)
    N = X.shape[0]
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build pairs of 3×3 grids for Stage 3 (Δε) training.

    Each pair consists of two *adjacent interior* scan points A and B.
    The label is Δε = ε_B − ε_A (Voigt, 6 components).

    Both A and B must be interior points (full 3×3 neighbourhood available),
    so the usable range shrinks by one additional step in the pair direction.

    Args:
        X:          (N, H, W) or (N, 1, H, W) flat pattern array.
        y_strain:   (N, 6) absolute Voigt strain labels.
        grid_rows:  Number of scan rows.
        grid_cols:  Number of scan columns.
        directions: Which adjacency directions to include.
                    'horizontal' → A=(r,c), B=(r,c+1)
                    'vertical'   → A=(r,c), B=(r+1,c)

    Returns:
        grids_a:      (M, 9, H, W)
        grids_b:      (M, 9, H, W)
        delta_strain: (M, 6)   Δε = ε_B − ε_A
        pos_a:        (M, 2)   [row, col] of center of grid A
        pos_b:        (M, 2)   [row, col] of center of grid B
    """
    X, H, W = _squeeze_channel(X)
    N = X.shape[0]
    if N != grid_rows * grid_cols:
        raise ValueError(
            f"X has {N} patterns but grid_rows*grid_cols = {grid_rows * grid_cols}."
        )

    X_2d = X.reshape(grid_rows, grid_cols, H, W)
    y_2d = y_strain.reshape(grid_rows, grid_cols, 6)

    ga_list:  list[np.ndarray] = []
    gb_list:  list[np.ndarray] = []
    dy_list:  list[np.ndarray] = []
    pa_list:  list[np.ndarray] = []  # position of center A  [row, col]
    pb_list:  list[np.ndarray] = []  # position of center B  [row, col]

    if "horizontal" in directions:
        # A=(r,c), B=(r,c+1); both need full 3×3 neighbourhoods.
        # A: c ∈ [1, cols-2],  B: c+1 ∈ [1, cols-2] → c ∈ [1, cols-3]
        for r in range(1, grid_rows - 1):
            for c in range(1, grid_cols - 2):
                ga_list.append(X_2d[r - 1 : r + 2, c - 1 : c + 2].reshape(9, H, W))
                gb_list.append(X_2d[r - 1 : r + 2, c : c + 3].reshape(9, H, W))
                dy_list.append(y_2d[r, c + 1] - y_2d[r, c])
                pa_list.append(np.array([r, c],     dtype=np.int32))
                pb_list.append(np.array([r, c + 1], dtype=np.int32))

    if "vertical" in directions:
        # A=(r,c), B=(r+1,c); both need full 3×3 neighbourhoods.
        # A: r ∈ [1, rows-2],  B: r+1 ∈ [1, rows-2] → r ∈ [1, rows-3]
        for r in range(1, grid_rows - 2):
            for c in range(1, grid_cols - 1):
                ga_list.append(X_2d[r - 1 : r + 2, c - 1 : c + 2].reshape(9, H, W))
                gb_list.append(X_2d[r : r + 3, c - 1 : c + 2].reshape(9, H, W))
                dy_list.append(y_2d[r + 1, c] - y_2d[r, c])
                pa_list.append(np.array([r,     c], dtype=np.int32))
                pb_list.append(np.array([r + 1, c], dtype=np.int32))

    if not ga_list:
        raise ValueError(
            f"No pairs found. Check grid dimensions ({grid_rows}×{grid_cols}) "
            f"and directions {directions}."
        )

    return (
        np.stack(ga_list),
        np.stack(gb_list),
        np.stack(dy_list),
        np.stack(pa_list),
        np.stack(pb_list),
    )


def build_grid_index(
    grid_rows: int,
    grid_cols: int,
) -> np.ndarray:
    """
    Return flat scan indices of all interior 3×3 grid centres.

    Interior means every point that has a full ring of 8 neighbours,
    i.e. rows 1..grid_rows-2, cols 1..grid_cols-2.

    Returns:
        center_idx: (M,) int32 flat scan indices,
                    M = (grid_rows-2) * (grid_cols-2)
    """
    centers = []
    for r in range(1, grid_rows - 1):
        for c in range(1, grid_cols - 1):
            centers.append(r * grid_cols + c)
    if not centers:
        raise ValueError(
            f"No interior points in a {grid_rows}×{grid_cols} grid. "
            "Need at least 3 rows and 3 columns."
        )
    return np.array(centers, dtype=np.int32)


def build_pair_index(
    grid_rows: int,
    grid_cols: int,
    directions: tuple[str, ...] = ("horizontal", "vertical"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return pair index arrays without touching any pattern data.

    Each pair consists of two adjacent interior scan centres A and B.
    Both centres must have a full 3×3 neighbourhood available, so the
    usable range shrinks by one step in the pair direction compared to
    build_grid_index.

    Returns:
        idx_a:  (M,) int32 flat scan index of centre A
        idx_b:  (M,) int32 flat scan index of centre B
        pos_a:  (M, 2) int32 [row, col] of centre A
        pos_b:  (M, 2) int32 [row, col] of centre B
    """
    idx_a_list: list[int] = []
    idx_b_list: list[int] = []
    pa_list:    list[list[int]] = []
    pb_list:    list[list[int]] = []

    if "horizontal" in directions:
        # A=(r,c), B=(r,c+1); c ∈ [1, cols-3] so both have full 3×3 neighbourhoods
        for r in range(1, grid_rows - 1):
            for c in range(1, grid_cols - 2):
                idx_a_list.append(r * grid_cols + c)
                idx_b_list.append(r * grid_cols + c + 1)
                pa_list.append([r, c])
                pb_list.append([r, c + 1])

    if "vertical" in directions:
        # A=(r,c), B=(r+1,c); r ∈ [1, rows-3]
        for r in range(1, grid_rows - 2):
            for c in range(1, grid_cols - 1):
                idx_a_list.append(r * grid_cols + c)
                idx_b_list.append((r + 1) * grid_cols + c)
                pa_list.append([r, c])
                pb_list.append([r + 1, c])

    if not idx_a_list:
        raise ValueError(
            f"No pairs found. Check grid dimensions ({grid_rows}×{grid_cols}) "
            f"and directions {directions}."
        )
    return (
        np.array(idx_a_list, dtype=np.int32),
        np.array(idx_b_list, dtype=np.int32),
        np.array(pa_list,    dtype=np.int32),
        np.array(pb_list,    dtype=np.int32),
    )


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
    Pairwise 3×3 grid dataset for Stage 3. Returns
    (grid_a, grid_b, targets_dict, pos_a, pos_b).

    pos_a / pos_b are integer [row, col] tensors of shape (2,) giving the
    scan-grid position of the center pattern in each 3×3 neighbourhood.
    They are used by the Saint-Venant loop-consistency loss to identify
    which pairs in a batch form closed rectangular loops.

    Grid patterns are ordered row-major (index 4 = center):
        0 1 2
        3 4 5
        6 7 8

    Args:
        grids_a, grids_b: (N, 9, H, W).
        targets:          Dict of label arrays, typically {'strain': (N,6)} for Δε.
        pos_a, pos_b:     (N, 2) int32 arrays of scan-grid positions.
                          Pass None if position metadata is unavailable.
        stats:            Normalisation stats from training split.
                          If None, computed from both grids — only pass None for training.
        norm_method:      'minmax' or 'zscore'.
    """

    def __init__(
        self,
        grids_a: np.ndarray,
        grids_b: np.ndarray,
        targets: dict[str, np.ndarray],
        pos_a: np.ndarray | None = None,
        pos_b: np.ndarray | None = None,
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

        N = len(grids_a)
        if pos_a is not None:
            self.pos_a = torch.from_numpy(pos_a.astype(np.int32))  # (N, 2)
            self.pos_b = torch.from_numpy(pos_b.astype(np.int32))  # (N, 2)
        else:
            # Fallback: all positions unknown — loop-consistency loss disabled.
            self.pos_a = torch.full((N, 2), -1, dtype=torch.int32)
            self.pos_b = torch.full((N, 2), -1, dtype=torch.int32)

    def __len__(self) -> int:
        return len(self.grids_a)

    def __getitem__(self, idx: int):
        return (
            self.grids_a[idx],
            self.grids_b[idx],
            {k: v[idx] for k, v in self.targets.items()},
            self.pos_a[idx],
            self.pos_b[idx],
        )


# ---------------------------------------------------------------------------
# Memory-efficient lazy datasets (on-the-fly patch extraction)
# ---------------------------------------------------------------------------


class LazyGridDataset(Dataset):
    """
    Stage 2 grid dataset that extracts 3×3 patches on-the-fly.

    Unlike GridDataset, this class stores only the original flat X array
    and a list of centre flat-indices, avoiding the O(M × 9 × H × W)
    pre-allocation that OOM-kills the process for large scans.

    Peak memory ≈ O(N × H × W) [just X] + O(batch × 9 × H × W) [one batch].

    Pattern grid ordering (row-major, index 4 = centre):
        0 1 2
        3 4 5
        6 7 8

    Args:
        X:           (N, H, W) or (N, 1, H, W) float32 patterns.
                     May be a memory-mapped numpy array for extra savings.
        grid_rows:   Number of scan rows.
        grid_cols:   Number of scan columns.
        center_idx:  (M,) int32 flat scan indices of centres for this split.
                     Obtain with build_grid_index()[split_indices].
        targets:     Dict of (M, *) float64/float32 label arrays (pre-indexed
                     to match center_idx).
        stats:       Normalisation stats dict from compute_norm_stats().
                     Must not be None (compute from training split in the
                     training script and pass explicitly to val/test splits).
        norm_method: 'minmax' or 'zscore'.
    """

    def __init__(
        self,
        X: np.ndarray,
        grid_rows: int,
        grid_cols: int,
        center_idx: np.ndarray,
        targets: dict[str, np.ndarray],
        stats: dict,
        norm_method: str = "minmax",
    ):
        X_sq, H, W = _squeeze_channel(X)
        self.X = X_sq                  # (N, H, W) float32 — stored as reference
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.H = H
        self.W = W
        self.center_idx = center_idx   # (M,) int32
        self.stats = stats
        self.norm_method = norm_method
        self.targets = {k: torch.from_numpy(np.asarray(v, dtype=np.float32))
                        for k, v in targets.items()}

    def _extract_grid(self, center_flat: int) -> torch.Tensor:
        r = int(center_flat) // self.grid_cols
        c = int(center_flat) % self.grid_cols
        flat = [
            (r - 1) * self.grid_cols + (c - 1),
            (r - 1) * self.grid_cols + c,
            (r - 1) * self.grid_cols + (c + 1),
            r       * self.grid_cols + (c - 1),
            r       * self.grid_cols + c,
            r       * self.grid_cols + (c + 1),
            (r + 1) * self.grid_cols + (c - 1),
            (r + 1) * self.grid_cols + c,
            (r + 1) * self.grid_cols + (c + 1),
        ]
        patch = self.X[flat].astype(np.float32)          # (9, H, W) — new allocation
        patch = apply_norm(patch, self.stats, self.norm_method)
        return torch.tensor(patch[:, None], dtype=torch.float32)  # (9, 1, H, W)

    def __len__(self) -> int:
        return len(self.center_idx)

    def __getitem__(self, idx: int):
        grid = self._extract_grid(int(self.center_idx[idx]))
        return grid, {k: v[idx] for k, v in self.targets.items()}


class LazyGridPairDataset(Dataset):
    """
    Stage 3 pairwise dataset that extracts 3×3 patches on-the-fly.

    Stores only the original flat X array and pair index arrays — no
    pre-materialised grids — so memory scales as O(N × H × W) rather
    than O(M × 18 × H × W).

    Returns (grid_a, grid_b, targets_dict, pos_a, pos_b) per item,
    identical interface to GridPairDataset.

    Args:
        X:           (N, H, W) or (N, 1, H, W) float32 patterns.
        grid_rows:   Number of scan rows.
        grid_cols:   Number of scan columns.
        idx_a:       (M,) int32 flat scan indices of centre A.
        idx_b:       (M,) int32 flat scan indices of centre B.
        targets:     Dict of (M, *) arrays (e.g. {'strain': (M, 6)} Δε).
        pos_a:       (M, 2) int32 [row, col] of centre A. Pass None if
                     position metadata is unavailable.
        pos_b:       (M, 2) int32 [row, col] of centre B.
        stats:       Normalisation stats from compute_norm_stats() on the
                     training split. Must not be None.
        norm_method: 'minmax' or 'zscore'.
    """

    def __init__(
        self,
        X: np.ndarray,
        grid_rows: int,
        grid_cols: int,
        idx_a: np.ndarray,
        idx_b: np.ndarray,
        targets: dict[str, np.ndarray],
        pos_a: np.ndarray | None = None,
        pos_b: np.ndarray | None = None,
        stats: dict | None = None,
        norm_method: str = "minmax",
    ):
        X_sq, H, W = _squeeze_channel(X)
        self.X = X_sq
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.H = H
        self.W = W
        self.idx_a = idx_a
        self.idx_b = idx_b
        self.stats = stats
        self.norm_method = norm_method
        self.targets = {k: torch.from_numpy(np.asarray(v, dtype=np.float32))
                        for k, v in targets.items()}
        M = len(idx_a)
        if pos_a is not None and pos_b is not None:
            self.pos_a = torch.from_numpy(pos_a.astype(np.int32))
            self.pos_b = torch.from_numpy(pos_b.astype(np.int32))
        else:
            self.pos_a = torch.full((M, 2), -1, dtype=torch.int32)
            self.pos_b = torch.full((M, 2), -1, dtype=torch.int32)

    def _extract_grid(self, center_flat: int) -> torch.Tensor:
        r = int(center_flat) // self.grid_cols
        c = int(center_flat) % self.grid_cols
        flat = [
            (r - 1) * self.grid_cols + (c - 1),
            (r - 1) * self.grid_cols + c,
            (r - 1) * self.grid_cols + (c + 1),
            r       * self.grid_cols + (c - 1),
            r       * self.grid_cols + c,
            r       * self.grid_cols + (c + 1),
            (r + 1) * self.grid_cols + (c - 1),
            (r + 1) * self.grid_cols + c,
            (r + 1) * self.grid_cols + (c + 1),
        ]
        patch = self.X[flat].astype(np.float32)
        patch = apply_norm(patch, self.stats, self.norm_method)
        return torch.tensor(patch[:, None], dtype=torch.float32)  # (9, 1, H, W)

    def __len__(self) -> int:
        return len(self.idx_a)

    def __getitem__(self, idx: int):
        grid_a = self._extract_grid(int(self.idx_a[idx]))
        grid_b = self._extract_grid(int(self.idx_b[idx]))
        return (
            grid_a,
            grid_b,
            {k: v[idx] for k, v in self.targets.items()},
            self.pos_a[idx],
            self.pos_b[idx],
        )
