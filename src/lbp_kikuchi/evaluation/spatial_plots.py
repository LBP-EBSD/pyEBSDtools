"""
Spatial strain visualisations: Voigt components as (row, col) heatmaps or scatters.

Pair models predict Δε on edges. For **parity vs scan position**, use
``plot_pair_delta_vs_xy`` on the eval split (no extra inference). Optional
``reconstruct_epsilon_from_pair_deltas`` + heatmaps integrate Δε to ε on the full
grid (expensive; meant for qualitative use, not the same as per-pair Δε plots).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VOIGT_LABELS = ["ε11", "ε22", "ε33", "ε23", "ε13", "ε12"]
DELTA_LABELS = ["Δε11", "Δε22", "Δε33", "Δε23", "Δε13", "Δε12"]


def flat_scan_to_grid(y_flat: np.ndarray, grid_rows: int, grid_cols: int) -> np.ndarray:
    """(N, K) row-major scan → (grid_rows, grid_cols, K)."""
    return y_flat.reshape(grid_rows, grid_cols, y_flat.shape[-1])


def scatter_centres_to_grid(
    center_flat: np.ndarray,
    values: np.ndarray,
    grid_rows: int,
    grid_cols: int,
) -> np.ndarray:
    """Place (M, K) samples at centre flat indices; NaN elsewhere."""
    out = np.full((grid_rows, grid_cols, values.shape[1]), np.nan, dtype=np.float64)
    for k in range(len(center_flat)):
        f = int(center_flat[k])
        r, c = f // grid_cols, f % grid_cols
        out[r, c] = values[k]
    return out


def boundary_flat_indices(grid_rows: int, grid_cols: int) -> np.ndarray:
    """Flat indices of the outer ring of the scan grid."""
    idx: list[int] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            if r == 0 or r == grid_rows - 1 or c == 0 or c == grid_cols - 1:
                idx.append(r * grid_cols + c)
    return np.array(sorted(set(idx)), dtype=np.int64)


def reconstruct_epsilon_from_pair_deltas(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    delta_pred: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    boundary_eps_flat: np.ndarray,
) -> np.ndarray:
    """
    Per Voigt component: sparse LSQR solution ε such that ε[b]-ε[a] ≈ Δ_pred on edges.

    Pins **boundary scan pixels** to ``boundary_eps_flat`` (typically ground-truth ε).
    Interior-only adjacency lists leave the pure incidence matrix rank-deficient; pinning
    the outer ring restores a well-posed Dirichlet-style reconstruction suitable for
    qualitative spatial plots.

    Args:
        idx_a, idx_b: (M,) flat indices for each directed pair A→B.
        delta_pred:   (M, 6) physical Δε predictions.
        boundary_eps_flat: (N, 6) ε values used on the boundary ring (e.g. truth).

    Returns:
        epsilon_hat (grid_rows, grid_cols, 6).
    """
    try:
        import scipy.sparse as sp
        from scipy.sparse.linalg import lsqr
    except ImportError as e:
        raise ImportError(
            "Stage 3 spatial reconstruction needs scipy. Install: pip install scipy"
        ) from e

    N = grid_rows * grid_cols
    M = len(idx_a)
    boundary = boundary_flat_indices(grid_rows, grid_cols)
    n_b = len(boundary)
    if boundary_eps_flat.shape != (N, delta_pred.shape[1]):
        raise ValueError(
            f"boundary_eps_flat expected shape ({N}, 6), got {boundary_eps_flat.shape}"
        )

    eps_hat = np.zeros((grid_rows, grid_cols, 6), dtype=np.float64)

    for comp in range(6):
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs_list: list[float] = []

        for k in range(M):
            ia = int(idx_a[k])
            ib = int(idx_b[k])
            rk = k
            rows.extend([rk, rk])
            cols.extend([ia, ib])
            data.extend([-1.0, 1.0])
            rhs_list.append(delta_pred[k, comp])

        row_off = M
        for j, bi in enumerate(boundary):
            rows.append(row_off + j)
            cols.append(int(bi))
            data.append(1.0)
            rhs_list.append(boundary_eps_flat[int(bi), comp])

        n_rows = M + n_b
        A = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, N))
        b = np.asarray(rhs_list, dtype=np.float64)

        sol = lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=max(10 * N, 20000))[0]
        eps_hat[..., comp] = sol.reshape(grid_rows, grid_cols)

    return eps_hat.astype(np.float32)


def _shared_color_limits(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Symmetric limits around 0 when both finite; else robust min/max."""
    am = np.nanmax(np.abs(a))
    bm = np.nanmax(np.abs(b))
    if np.isfinite(am) and np.isfinite(bm) and am > 0 and bm > 0:
        lo = -max(am, bm)
        hi = max(am, bm)
        return lo, hi
    vals = np.concatenate([a[np.isfinite(a)].ravel(), b[np.isfinite(b)].ravel()])
    if vals.size == 0:
        return -1e-6, 1e-6
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    pad = (hi - lo) * 0.05 or 1e-9
    return lo - pad, hi + pad


def plot_strain_maps_three_panel(
    true_grid: np.ndarray,
    pred_grid: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
    cmap: str = "RdBu_r",
) -> None:
    """
    One row per Voigt component; columns: True | Predicted | |error|.

    true_grid, pred_grid: (grid_rows, grid_cols, 6).
    """
    n_comp = len(labels)
    fig, axes = plt.subplots(n_comp, 3, figsize=(11, 2.4 * n_comp), squeeze=False)

    err_grid = np.abs(pred_grid - true_grid)

    for i, lbl in enumerate(labels):
        t = true_grid[..., i]
        p = pred_grid[..., i]
        e = err_grid[..., i]
        vmin, vmax = _shared_color_limits(t, p)

        for ax, arr, cbar_label in zip(
            axes[i],
            (t, p, e),
            ("True", "Predicted", "|error|"),
        ):
            if cbar_label == "|error|":
                im = ax.imshow(arr, origin="lower", aspect="auto", cmap="magma", vmin=0)
            else:
                im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{lbl} — {cbar_label}", fontsize=9)
            ax.set_xlabel("col")
            if i == n_comp // 2:
                ax.set_ylabel("row")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_edge_midpoints(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    labels: list[str],
    title_prefix: str,
    out_prefix: Path,
    cmap: str = "RdBu_r",
) -> None:
    """
    Scatter Δε at edge midpoints (fractional row/col) for true vs pred.

    idx_a, idx_b: flat indices; delta_* : (M, 6).

    Writes ``out_prefix`` + ``_delta_true.png`` and ``_delta_pred.png``.
    """
    ra = idx_a.astype(np.float64) // grid_cols
    ca = idx_a.astype(np.float64) % grid_cols
    rb = idx_b.astype(np.float64) // grid_cols
    cb = idx_b.astype(np.float64) % grid_cols
    mr = (ra + rb) / 2.0
    mc = (ca + cb) / 2.0

    for suffix, values, sub in (
        ("delta_true", delta_true, "true Δε"),
        ("delta_pred", delta_pred, "predicted Δε"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        axes = axes.flatten()
        for k, lbl in enumerate(labels):
            ax = axes[k]
            v = values[:, k]
            vmin, vmax = float(v.min()), float(v.max())
            pad = (vmax - vmin) * 0.05 or 1e-9
            sc = ax.scatter(
                mc, mr, c=v, s=6, cmap=cmap, vmin=vmin - pad, vmax=vmax + pad, alpha=0.85
            )
            ax.set_title(f"{lbl} ({sub})", fontsize=10)
            ax.set_xlim(-0.5, grid_cols - 0.5)
            ax.set_ylim(-0.5, grid_rows - 0.5)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            plt.colorbar(sc, ax=ax, fraction=0.046)
        fig.suptitle(f"{title_prefix} — {sub} @ edge midpoints", fontsize=12)
        plt.tight_layout()
        out_path = Path(str(out_prefix) + f"_{suffix}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Spatial Δε plot → {out_path}")


def plot_pair_delta_vs_xy(
    pos_mid: np.ndarray,
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    labels: list[str],
    title: str,
    out_path: Path,
    cmap: str = "RdBu_r",
) -> None:
    """
    Scatter Δε vs scan position for the **evaluated pairs only** (one point per pair).

    ``pos_mid`` is (N, 2) with ``[row, col]`` at the edge midpoint (same convention
    as ``build_pair_index`` centres). X axis = col, Y axis = row (y inverted like
    image coordinates).

    This uses only tensors already produced in the eval DataLoader — no extra
    inference. Compare to ``plot_strain_maps_three_panel`` on LSQR-reconstructed ε,
    which is a different quantity and requires a global solve.
    """
    if pos_mid.shape[0] != delta_true.shape[0] or pos_mid.shape[0] != delta_pred.shape[0]:
        raise ValueError(
            f"Length mismatch: pos_mid {pos_mid.shape[0]}, "
            f"delta_true {delta_true.shape[0]}, delta_pred {delta_pred.shape[0]}"
        )
    mr = pos_mid[:, 0]
    mc = pos_mid[:, 1]
    n_comp = len(labels)
    fig, axes = plt.subplots(n_comp, 2, figsize=(10, 2.2 * n_comp), squeeze=False)

    for i, lbl in enumerate(labels):
        t = delta_true[:, i]
        p = delta_pred[:, i]
        vmin, vmax = _shared_color_limits(t, p)

        for ax, arr, name in zip(axes[i], (t, p), ("true Δε", "predicted Δε")):
            sc = ax.scatter(
                mc,
                mr,
                c=arr,
                s=14,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.9,
                linewidths=0,
            )
            ax.set_title(f"{lbl} — {name}", fontsize=9)
            ax.set_xlim(-0.5, grid_cols - 0.5)
            ax.set_ylim(-0.5, grid_rows - 0.5)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xlabel("col")
            if i == n_comp // 2:
                ax.set_ylabel("row")
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Spatial Δε vs (row,col) → {out_path}")
