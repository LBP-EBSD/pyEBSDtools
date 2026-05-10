"""
Evaluation script for all training stages.

Automatically detects whether a run directory contains a Stage 1
(single-pattern), Stage 2 (grid), or Stage 3 (pair) checkpoint and
runs the appropriate evaluator.

Usage (from repo root):
    # Evaluate the most-recent run (any stage):
    python scripts/infer_eval.py

    # Evaluate a specific run:
    python scripts/infer_eval.py --run-dir outputs/2026-05-10/15-35-31

    # Choose which split to evaluate:
    python scripts/infer_eval.py --split val      # default
    python scripts/infer_eval.py --split train
    python scripts/infer_eval.py --split test
    python scripts/infer_eval.py --split all

    # Other options:
    python scripts/infer_eval.py --checkpoint last.pt
    python scripts/infer_eval.py --batch-size 64
    python scripts/infer_eval.py --max-samples 2000
    python scripts/infer_eval.py --full-spatial-viz   # optional second pass + LSQR maps
    python scripts/infer_eval.py --no-plot

Outputs (written to the run directory):
    eval_scatter_<split>.png          parity plots for current split
    eval_results_<split>.json
    eval_spatial_stage3_delta_vs_xy_<split>.png   Stage 3: Δε vs (col,row), eval split only
    eval_spatial_*.png                full-scan heatmaps / LSQR ε only with --full-spatial-viz

Note:
    * TensorBoard ``strain_mae`` / ``strain_rmse`` are on **normalised** targets.
      ``infer_eval`` scatter/maps use **physical** strain — trends align; numeric scales differ.

Scaling (all stages):
    * ``norm_stats.json`` stores input stats (min/max or mean/std) from **training
      data only** — Stage 1 uses ``X[train_idx]``; Stages 2–3 use every scan pixel
      that appears in any **training** 3×3 patch (union around train grid centres
      or pair centres). Val/test pixels are normalised with those stats (values may
      fall outside [0, 1] for min–max).
    * ``y_mean`` / ``y_std`` are from the **training split only** (per Voigt component).
    * The network is trained on **normalised** targets; metrics in ``infer_eval``
      and scatter axes are **physical** strain (ε or Δε) after
      ``pred_phys = pred_norm * y_std + y_mean``.
    * MAE is ``mean(|pred − true|)`` over all points — often well below the
      **maximum** vertical distance in a parity plot when many true values sit
      near zero (e.g. collapsed or small-|Δε| bulk); subplot titles also show
      ``max|e|`` for that component.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from lbp_kikuchi.data.dataset import (
    apply_norm,
    build_grid_index,
    build_pair_index,
    LazyGridDataset,
    LazyGridPairDataset,
)
from lbp_kikuchi.evaluation.spatial_plots import (
    flat_scan_to_grid,
    plot_delta_edge_midpoints,
    plot_pair_delta_vs_xy,
    plot_strain_maps_three_panel,
    reconstruct_epsilon_from_pair_deltas,
    scatter_centres_to_grid,
)

VOIGT_LABELS = ["ε11", "ε22", "ε33", "ε23", "ε13", "ε12"]
DELTA_LABELS = ["Δε11", "Δε22", "Δε33", "Δε23", "Δε13", "Δε12"]


# ── Run helpers ───────────────────────────────────────────────────────────────

def find_latest_run(outputs_root: Path) -> Path:
    candidates = sorted(
        (p for p in outputs_root.glob("*/*") if (p / "checkpoints").is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No completed run directories found under {outputs_root}."
        )
    return candidates[-1]


def load_run(run_dir: Path, checkpoint_name: str = "best.pt") -> tuple:
    config_path = run_dir / "config_snapshot.json"
    norm_path   = run_dir / "checkpoints" / "norm_stats.json"
    ckpt_path   = run_dir / "checkpoints" / checkpoint_name

    for p in (config_path, norm_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    cfg          = json.loads(config_path.read_text())
    norm_payload = json.loads(norm_path.read_text())
    norm_method  = norm_payload.pop("norm_method")
    norm_stats   = norm_payload

    missing = [k for k in ("y_mean", "y_std") if k not in norm_stats]
    if missing:
        raise KeyError(
            f"Missing target normalisation keys {missing} in {norm_path}. "
            "Re-train with the current pipeline to use this evaluator."
        )

    return cfg, norm_method, norm_stats, ckpt_path


def is_pair_run(cfg: dict) -> bool:
    """Detect Stage 3 (pair model) from the saved config."""
    return "directions" in cfg.get("data", {})


def is_grid_run(cfg: dict) -> bool:
    """Detect Stage 2 (grid model) from the saved config."""
    return (
        "grid_rows" in cfg.get("data", {})
        and "directions" not in cfg.get("data", {})
    )


def select_split_indices(run_dir: Path, split: str, n_total: int) -> np.ndarray:
    if split == "all":
        return np.arange(n_total, dtype=np.int64)

    split_path = run_dir / "checkpoints" / "split_indices.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split index file not found: {split_path}. Use --split all."
        )
    payload = json.loads(split_path.read_text())
    key = f"{split}_idx"
    if key not in payload:
        raise KeyError(
            f"Key '{key}' not found in split file. Available: {list(payload.keys())}"
        )
    idx = np.array(payload[key], dtype=np.int64)
    if idx.size == 0:
        raise ValueError(f"Split '{split}' is empty.")
    return idx


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_scatter(
    preds: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()
    err = np.abs(preds - targets)
    overall_mae = float(err.mean())
    n = len(preds)

    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        p_full = preds[:, i]
        t_full = targets[:, i]
        mae = float(np.abs(p_full - t_full).mean())
        rmse = float(np.sqrt(((p_full - t_full) ** 2).mean()))
        max_e = float(np.abs(p_full - t_full).max())

        p, t = p_full, t_full
        if len(p) > 2000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(p), 2000, replace=False)
            p, t = p[idx], t[idx]

        lo = min(p.min(), t.min())
        hi = max(p.max(), t.max())
        pad = (hi - lo) * 0.05 or 1e-6

        ax.scatter(t, p, s=5, alpha=0.35, linewidths=0, color="steelblue")
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.2, label="ideal")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("True", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(
            f"{lbl}  MAE={mae:.4g}  RMSE={rmse:.4g}  max|e|={max_e:.4g}",
            fontsize=10,
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"{title}   N={n}   overall MAE={overall_mae:.4g}  (mean |err|; see max|e| per panel)",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot  → {out_path}")


def print_table(preds: np.ndarray, targets: np.ndarray,
                labels: list[str], split: str) -> dict:
    errors = np.abs(preds - targets)
    overall_mae  = float(errors.mean())
    overall_rmse = float(np.sqrt(((preds - targets) ** 2).mean()))
    max_err      = float(errors.max())
    per_comp     = {lbl: float(errors[:, i].mean()) for i, lbl in enumerate(labels)}
    per_comp_max = {lbl: float(errors[:, i].max()) for i, lbl in enumerate(labels)}
    max_bar      = max(per_comp.values()) or 1e-9

    print(f"\n{'='*60}")
    print(f"  Evaluation — split={split}  N={len(preds)}")
    print(f"{'='*60}")
    print(f"  {'Overall MAE':<20} {overall_mae:>12.6f}")
    print(f"  {'Overall RMSE':<20} {overall_rmse:>12.6f}")
    print(f"  {'Max |error| (any)':<20} {max_err:>12.6f}")
    print(f"\n  Per-component MAE / max |error|:")
    print(f"  {'Component':<12} {'MAE':>10}  {'max|e|':>10}  bar")
    print(f"  {'-'*56}")
    for lbl in labels:
        mae = per_comp[lbl]
        mx = per_comp_max[lbl]
        bar = "█" * int(mae / max_bar * 20 + 0.5)
        print(f"  {lbl:<12} {mae:>10.6f}  {mx:>10.6f}  {bar}")
    print(f"{'='*60}\n")

    return {
        "n_samples": len(preds),
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "max_abs_error": max_err,
        "per_component_mae": per_comp,
        "per_component_max_abs_error": per_comp_max,
    }


def resolve_scan_grid(cfg: dict, n_patterns: int) -> tuple[int, int] | None:
    """Infer (grid_rows, grid_cols) from config or square root of N."""
    data = cfg.get("data", {})
    dr, dc = data.get("grid_rows"), data.get("grid_cols")
    if dr is not None and dc is not None:
        dr_i, dc_i = int(dr), int(dc)
        if dr_i * dc_i == n_patterns:
            return dr_i, dc_i
        print(
            f"[spatial viz] grid_rows×grid_cols ({dr_i}×{dc_i}) ≠ N={n_patterns}; "
            "will try √N fallback."
        )
    root = int(np.sqrt(n_patterns))
    if root * root == n_patterns:
        print(f"[spatial viz] Using inferred square grid {root}×{root}.")
        return root, root
    return None


def _input_stats(norm_stats: dict) -> dict:
    return {k: v for k, v in norm_stats.items() if k not in ("y_mean", "y_std")}


def spatial_viz_stage1(
    args,
    cfg: dict,
    norm_method: str,
    norm_stats: dict,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    data_dir = args.data_dir or Path(cfg["data"]["path"])
    X = np.load(data_dir / args.patterns_file)
    y_strain = np.load(data_dir / args.strain_file)
    if len(X) != len(y_strain):
        raise ValueError("X and y_strain length mismatch.")

    grid = resolve_scan_grid(cfg, len(X))
    if grid is None:
        raise ValueError(
            "Cannot infer scan grid for spatial maps. "
            "Set data.grid_rows and data.grid_cols in config or use N=m² patterns."
        )
    grid_rows, grid_cols = grid

    X_sq = X[:, 0] if X.ndim == 4 else X
    y_mean = np.array(norm_stats["y_mean"])
    y_std = np.array(norm_stats["y_std"])

    n_pat = len(X_sq)
    print(
        f"\n[spatial viz] Stage 1 — full scan: {n_pat} patterns in "
        f"{(n_pat + args.batch_size - 1) // args.batch_size} batches "
        f"(use --no-spatial-viz to skip).\n",
        flush=True,
    )
    all_preds: list[np.ndarray] = []
    for start in range(0, len(X_sq), args.batch_size):
        xb = X_sq[start : start + args.batch_size].astype(np.float32)
        xb = apply_norm(xb, norm_stats, norm_method)
        if xb.ndim == 3:
            xb = xb[:, np.newaxis]
        with torch.no_grad():
            p = model(torch.from_numpy(xb).to(device))["strain"].cpu().numpy()
        all_preds.append(p)
    preds_flat = np.concatenate(all_preds) * y_std + y_mean

    true_grid = flat_scan_to_grid(y_strain.astype(np.float64), grid_rows, grid_cols)
    pred_grid = flat_scan_to_grid(preds_flat.astype(np.float64), grid_rows, grid_cols)

    out = args.run_dir / f"eval_spatial_stage1_eps_maps_full.png"
    plot_strain_maps_three_panel(
        true_grid.astype(np.float32),
        pred_grid.astype(np.float32),
        VOIGT_LABELS,
        f"Stage 1 — ε on scan grid (full data, split={args.split!r} parity elsewhere)",
        out,
    )
    print(f"Spatial strain maps → {out}")


def spatial_viz_grid(
    args,
    cfg: dict,
    norm_method: str,
    norm_stats: dict,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    from torch.utils.data import DataLoader

    data_dir = args.data_dir or Path(cfg["data"]["path"])
    X = np.load(data_dir / cfg["data"].get("patterns_file", "X_patterns.npy"))
    y_strain = np.load(data_dir / cfg["data"].get("strain_file", "y_strain.npy"))
    grid_rows = int(cfg["data"]["grid_rows"])
    grid_cols = int(cfg["data"]["grid_cols"])
    img_size = cfg["model"].get("img_size", 224)

    center_idx = build_grid_index(grid_rows, grid_cols)
    y_grids = y_strain[center_idx]
    y_mean = np.array(norm_stats["y_mean"])
    y_std = np.array(norm_stats["y_std"])
    y_norm = (y_grids - y_mean) / y_std

    n_cent = len(center_idx)
    print(
        f"\n[spatial viz] Stage 2 — all interior centres: {n_cent} samples "
        f"({(n_cent + args.batch_size - 1) // args.batch_size} batches; "
        f"--no-spatial-viz to skip).\n",
        flush=True,
    )

    X_sq = X[:, 0] if X.ndim == 4 else X
    input_stats = _input_stats(norm_stats)

    ds = LazyGridDataset(
        X_sq,
        grid_rows,
        grid_cols,
        center_idx,
        {"strain": y_norm},
        stats=input_stats,
        norm_method=norm_method,
        img_size=img_size,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, cfg["training"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for grid, _ in loader:
            chunks.append(model(grid.to(device))["strain"].cpu().numpy())
    preds_norm = np.concatenate(chunks)
    preds_interior = preds_norm * y_std + y_mean

    true_interior = y_grids.astype(np.float64)
    pred_grid = scatter_centres_to_grid(center_idx, preds_interior, grid_rows, grid_cols)
    true_grid = scatter_centres_to_grid(center_idx, true_interior, grid_rows, grid_cols)

    out = args.run_dir / "eval_spatial_stage2_eps_interior_full.png"
    plot_strain_maps_three_panel(
        true_grid.astype(np.float32),
        pred_grid.astype(np.float32),
        VOIGT_LABELS,
        f"Stage 2 — ε at interior 3×3 centres (NaN outside); split={args.split!r} for scatter only",
        out,
    )
    print(f"Spatial strain maps → {out}")


def spatial_viz_pair(
    args,
    cfg: dict,
    norm_method: str,
    norm_stats: dict,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    from torch.utils.data import DataLoader

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(it, **_kw):
            return it

    data_dir = args.data_dir or Path(cfg["data"]["path"])
    X = np.load(data_dir / cfg["data"].get("patterns_file", "X_patterns.npy"))
    y_strain = np.load(data_dir / cfg["data"].get("strain_file", "y_strain.npy"))
    grid_rows = int(cfg["data"]["grid_rows"])
    grid_cols = int(cfg["data"]["grid_cols"])
    directions = tuple(cfg["data"]["directions"])
    img_size = cfg["model"].get("img_size", 224)

    idx_a, idx_b, _, _ = build_pair_index(grid_rows, grid_cols, directions=directions)
    delta_strain = y_strain[idx_b] - y_strain[idx_a]
    n_pairs = len(idx_a)
    n_batches = (n_pairs + args.batch_size - 1) // args.batch_size
    print(
        f"\n[spatial viz] Stage 3 — second GPU pass: ALL {n_pairs} neighbor pairs "
        f"({n_batches} batches) for LSQR ε maps + dense midpoint Δε figures. "
        f"(Default eval already wrote Δε vs (row,col) for the split only.) "
        f"Next time omit --full-spatial-viz.\n",
        flush=True,
    )

    y_mean = np.array(norm_stats["y_mean"])
    y_std = np.array(norm_stats["y_std"])
    delta_norm = (delta_strain - y_mean) / y_std

    X_sq = X[:, 0] if X.ndim == 4 else X
    input_stats = _input_stats(norm_stats)

    ds = LazyGridPairDataset(
        X_sq,
        grid_rows,
        grid_cols,
        idx_a,
        idx_b,
        {"strain": delta_norm},
        stats=input_stats,
        norm_method=norm_method,
        img_size=img_size,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, cfg["training"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for grid_a, grid_b, _, _, _ in tqdm(
            loader,
            desc="Spatial viz infer",
            unit="batch",
            leave=True,
        ):
            chunks.append(
                model(grid_a.to(device), grid_b.to(device))["strain"].cpu().numpy()
            )
    preds_phys = np.concatenate(chunks) * y_std + y_mean

    print("[spatial viz] LSQR reconstructing ε from predicted Δε …", flush=True)
    eps_hat = reconstruct_epsilon_from_pair_deltas(
        idx_a,
        idx_b,
        preds_phys.astype(np.float64),
        grid_rows,
        grid_cols,
        boundary_eps_flat=y_strain.astype(np.float64),
    )

    true_full = flat_scan_to_grid(y_strain.astype(np.float64), grid_rows, grid_cols)

    out_maps = args.run_dir / "eval_spatial_stage3_eps_lsqr_full.png"
    plot_strain_maps_three_panel(
        true_full.astype(np.float32),
        eps_hat,
        VOIGT_LABELS,
        "Stage 3 — ε: true vs LSQR from predicted Δε (boundary ring fixed to truth)",
        out_maps,
    )
    print(f"Spatial strain maps → {out_maps}")

    mid_prefix = args.run_dir / "eval_spatial_stage3_pair_midpoints"
    plot_delta_edge_midpoints(
        idx_a,
        idx_b,
        delta_strain.astype(np.float64),
        preds_phys.astype(np.float64),
        grid_rows,
        grid_cols,
        DELTA_LABELS,
        f"Stage 3 pair — full edge list ({len(idx_a)} edges)",
        mid_prefix,
    )


# ── Stage 1: single-pattern evaluator ────────────────────────────────────────

def eval_single(args, cfg, norm_method, norm_stats, ckpt_path, device):
    from lbp_kikuchi.models.single_model import SinglePatternModel

    img_size = cfg["model"].get("img_size", 224)
    model = SinglePatternModel(
        feature_dim=cfg["model"]["feature_dim"],
        predict_orientation=cfg["training"].get("predict_orientation", False),
        img_size=img_size,
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()

    data_dir  = args.data_dir or Path(cfg["data"]["path"])
    X         = np.load(data_dir / args.patterns_file)
    y_strain  = np.load(data_dir / args.strain_file)
    print(f"Patterns      : {X.shape}  dtype={X.dtype}")

    split_idx = select_split_indices(args.run_dir, args.split, len(X))
    X_eval    = X[split_idx]
    y_eval    = y_strain[split_idx]
    print(f"Eval split    : {args.split}  n={len(X_eval)}")

    y_mean = np.array(norm_stats["y_mean"])
    y_std  = np.array(norm_stats["y_std"])

    all_preds = []
    for start in range(0, len(X_eval), args.batch_size):
        xb = X_eval[start : start + args.batch_size].astype(np.float32)
        xb = apply_norm(xb, norm_stats, norm_method)
        if xb.ndim == 3:
            xb = xb[:, np.newaxis]
        with torch.no_grad():
            p = model(torch.from_numpy(xb).to(device))["strain"].cpu().numpy()
        all_preds.append(p)

    preds   = np.concatenate(all_preds) * y_std + y_mean
    if args.max_samples is not None:
        n = min(args.max_samples, len(preds))
        preds, y_eval = preds[:n], y_eval[:n]
    results = print_table(preds, y_eval, VOIGT_LABELS, args.split)

    if not args.no_plot:
        plot_scatter(
            preds, y_eval, VOIGT_LABELS,
            f"Stage 1 — Voigt Strain Prediction  [{args.split}]",
            args.run_dir / f"eval_scatter_{args.split}.png",
        )
        if args.full_spatial_viz and not args.no_spatial_viz:
            try:
                spatial_viz_stage1(args, cfg, norm_method, norm_stats, model, device)
            except Exception as e:
                print(f"[spatial viz] Stage 1 skipped: {e}")
    return results


# ── Stage 2: grid evaluator (split indices index *grid samples*, not flat X) ──

def eval_grid(args, cfg, norm_method, norm_stats, ckpt_path, device):
    from lbp_kikuchi.models.grid_model import GridModel
    from torch.utils.data import DataLoader

    img_size = cfg["model"].get("img_size", 224)
    model = GridModel(
        feature_dim=cfg["model"]["feature_dim"],
        img_size=img_size,
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()

    data_dir = args.data_dir or Path(cfg["data"]["path"])
    X = np.load(data_dir / cfg["data"].get("patterns_file", "X_patterns.npy"))
    y_strain = np.load(data_dir / cfg["data"].get("strain_file", "y_strain.npy"))
    print(f"Patterns      : {X.shape}  dtype={X.dtype}")

    grid_rows = int(cfg["data"]["grid_rows"])
    grid_cols = int(cfg["data"]["grid_cols"])
    center_idx = build_grid_index(grid_rows, grid_cols)
    M = len(center_idx)
    y_grids = y_strain[center_idx]
    print(f"Grid samples  : {M}  (interior 3×3 centres)")

    split_idx = select_split_indices(args.run_dir, args.split, M)
    print(f"Eval split    : {args.split}  n={len(split_idx)}")

    y_mean = np.array(norm_stats["y_mean"])
    y_std = np.array(norm_stats["y_std"])
    y_norm = (y_grids - y_mean) / y_std

    X_sq = X[:, 0] if X.ndim == 4 else X
    input_stats = {k: v for k, v in norm_stats.items() if k not in ("y_mean", "y_std")}

    ds = LazyGridDataset(
        X_sq,
        grid_rows,
        grid_cols,
        center_idx[split_idx],
        {"strain": y_norm[split_idx]},
        stats=input_stats,
        norm_method=norm_method,
        img_size=img_size,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    all_preds = []
    with torch.no_grad():
        for grid, _tgt in loader:
            p = model(grid.to(device))["strain"].cpu().numpy()
            all_preds.append(p)

    preds_norm = np.concatenate(all_preds)
    preds = preds_norm * y_std + y_mean
    targets = y_grids[split_idx]

    if args.max_samples is not None:
        n = min(args.max_samples, len(preds))
        preds, targets = preds[:n], targets[:n]

    results = print_table(preds, targets, VOIGT_LABELS, args.split)

    if not args.no_plot:
        plot_scatter(
            preds,
            targets,
            VOIGT_LABELS,
            f"Stage 2 — Voigt strain ε(center)  [{args.split}]",
            args.run_dir / f"eval_scatter_{args.split}.png",
        )
        if args.full_spatial_viz and not args.no_spatial_viz:
            try:
                spatial_viz_grid(args, cfg, norm_method, norm_stats, model, device)
            except Exception as e:
                print(f"[spatial viz] Stage 2 skipped: {e}")
    return results


# ── Stage 3: pair evaluator ───────────────────────────────────────────────────

def eval_pair(args, cfg, norm_method, norm_stats, ckpt_path, device):
    from lbp_kikuchi.models.pair_model import PairModel
    from torch.utils.data import DataLoader

    img_size = cfg["model"].get("img_size", 224)
    model = PairModel(
        feature_dim=cfg["model"]["feature_dim"],
        img_size=img_size,
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()

    data_dir = args.data_dir or Path(cfg["data"]["path"])
    X        = np.load(data_dir / cfg["data"].get("patterns_file", "X_patterns.npy"))
    y_strain = np.load(data_dir / cfg["data"].get("strain_file",   "y_strain.npy"))
    print(f"Patterns      : {X.shape}  dtype={X.dtype}")

    grid_rows  = int(cfg["data"]["grid_rows"])
    grid_cols  = int(cfg["data"]["grid_cols"])
    directions = tuple(cfg["data"]["directions"])
    print(f"Scan grid     : {grid_rows} × {grid_cols}   directions={directions}")

    idx_a, idx_b, pos_a, pos_b = build_pair_index(grid_rows, grid_cols, directions)
    delta_strain = y_strain[idx_b] - y_strain[idx_a]   # (M, 6)
    M = len(idx_a)
    print(f"Total pairs   : {M}")

    split_idx = select_split_indices(args.run_dir, args.split, M)
    print(f"Eval split    : {args.split}  n={len(split_idx)}")

    # Midpoint [row, col] of each pair edge (for Δε vs scan position; no extra inference).
    pos_mid = (
        pos_a[split_idx].astype(np.float64) + pos_b[split_idx].astype(np.float64)
    ) * 0.5

    y_mean = np.array(norm_stats["y_mean"])
    y_std  = np.array(norm_stats["y_std"])

    # Normalise Δε with the same train-split μ, σ saved in norm_stats (matches train_pair).
    delta_norm = (delta_strain - y_mean) / y_std
    X_sq = X[:, 0] if X.ndim == 4 else X   # (N, H, W)
    input_stats = {k: v for k, v in norm_stats.items() if k not in ("y_mean", "y_std")}

    ds = LazyGridPairDataset(
        X_sq, grid_rows, grid_cols,
        idx_a[split_idx], idx_b[split_idx],
        {"strain": delta_norm[split_idx]},
        pos_a=pos_a[split_idx], pos_b=pos_b[split_idx],
        stats=input_stats, norm_method=norm_method,
        img_size=img_size,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    all_preds = []
    with torch.no_grad():
        for grid_a, grid_b, _tgt, _pa, _pb in loader:
            p = model(grid_a.to(device), grid_b.to(device))["strain"].cpu().numpy()
            all_preds.append(p)

    preds_norm = np.concatenate(all_preds)               # z-scored Δε (training target space)
    preds      = preds_norm * y_std + y_mean             # physical Δε
    targets    = delta_strain[split_idx]                 # physical Δε

    if args.max_samples is not None:
        n = min(args.max_samples, len(preds))
        preds, targets = preds[:n], targets[:n]
        pos_mid = pos_mid[:n]

    results = print_table(preds, targets, DELTA_LABELS, args.split)

    if not args.no_plot:
        plot_scatter(
            preds, targets, DELTA_LABELS,
            f"Stage 3 — Relative Strain Prediction (Δε)  [{args.split}]",
            args.run_dir / f"eval_scatter_{args.split}.png",
        )
        if not args.no_spatial_viz:
            try:
                plot_pair_delta_vs_xy(
                    pos_mid,
                    targets,
                    preds,
                    grid_rows,
                    grid_cols,
                    DELTA_LABELS,
                    f"Stage 3 — Δε vs scan position (eval split n={len(preds)})  [{args.split}]",
                    args.run_dir / f"eval_spatial_stage3_delta_vs_xy_{args.split}.png",
                )
            except Exception as e:
                print(f"[spatial viz] Stage 3 Δε vs (row,col) skipped: {e}")
        if args.full_spatial_viz and not args.no_spatial_viz:
            try:
                spatial_viz_pair(args, cfg, norm_method, norm_stats, model, device)
            except Exception as e:
                print(f"[spatial viz] Stage 3 full-scan skipped: {e}")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir",      type=Path, default=None)
    parser.add_argument("--split",        choices=("val", "train", "test", "all"),
                        default="val")
    parser.add_argument("--data-dir",     type=Path, default=None)
    parser.add_argument("--patterns-file", default="X_patterns.npy")
    parser.add_argument("--strain-file",   default="y_strain.npy")
    parser.add_argument("--checkpoint",   default="best.pt")
    parser.add_argument("--batch-size",   type=int, default=64)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap samples for metrics/scatter and Δε-vs-xy plots (default: full split).",
    )
    parser.add_argument("--no-plot",      action="store_true")
    parser.add_argument(
        "--no-spatial-viz",
        action="store_true",
        help="Skip all optional spatial figures (Stage 3 Δε vs row/col; full-scan maps).",
    )
    parser.add_argument(
        "--full-spatial-viz",
        action="store_true",
        help=(
            "Run a second full-dataset inference + LSQR / dense maps (Stage 1–3). "
            "Heavy; default is off — eval split already covers metrics and Δε-vs-xy."
        ),
    )
    args = parser.parse_args()

    args.run_dir = args.run_dir or find_latest_run(Path("outputs"))
    print(f"Run dir       : {args.run_dir}")

    cfg, norm_method, norm_stats, ckpt_path = load_run(args.run_dir, args.checkpoint)
    print(f"Checkpoint    : {ckpt_path}")
    print(f"Norm method   : {norm_method}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}")

    if is_pair_run(cfg):
        print("Stage         : 3 — Pair model (Δε)")
        results = eval_pair(args, cfg, norm_method, norm_stats, ckpt_path, device)
    elif is_grid_run(cfg):
        print("Stage         : 2 — Grid model (ε)")
        results = eval_grid(args, cfg, norm_method, norm_stats, ckpt_path, device)
    else:
        print("Stage         : 1 — Single-pattern model (ε)")
        results = eval_single(args, cfg, norm_method, norm_stats, ckpt_path, device)

    out = args.run_dir / f"eval_results_{args.split}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Results       → {out}")


if __name__ == "__main__":
    main()
