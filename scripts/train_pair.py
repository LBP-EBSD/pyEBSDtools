"""
Stage 3 training: pair of 3×3 pattern grids → Δε (relative strain).

For each pair of adjacent scan points A and B, the model predicts:
    Δε = ε_B − ε_A

This is the physically meaningful learning target: strain is encoded as
relative pattern distortion between points, not absolute appearance.
The subtraction in PairModel (F_B − F_A) cancels shared bias and isolates
the deformation signal.

To reconstruct the absolute strain field from Δε predictions, accumulate
predictions along the scan grid (cumulative sum or least-squares integration).

Pair directions collected:
    horizontal  A=(r,c) → B=(r,c+1)
    vertical    A=(r,c) → B=(r+1,c)
Both directions give the model all finite-difference information needed to
reconstruct a 2D strain map.

Run from repo root:
    python scripts/train_pair.py
    python scripts/train_pair.py data.grid_rows=100 data.grid_cols=100
    python scripts/train_pair.py training.directions=[horizontal]

Config overrides (Hydra syntax):
    data.grid_rows=100
    data.grid_cols=100
    data.directions=[horizontal,vertical]
    training.epochs=50
    training.batch_size=8
    training.lr=5e-4
    model.feature_dim=128
    experiment_name=stage3_run1

Outputs (under outputs/YYYY-MM-DD/HH-MM-SS/):
    checkpoints/best.pt
    checkpoints/last.pt
    checkpoints/norm_stats.json
    checkpoints/split_indices.json
    config_snapshot.json
    metrics.csv / metrics.json
    tensorboard/
"""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from lbp_kikuchi.data.dataset import (
    LazyGridPairDataset,
    build_pair_index,
    compute_norm_stats,
)
from lbp_kikuchi.models.pair_model import PairModel
from lbp_kikuchi.training.engine import evaluate_pair, train_one_epoch_pair
from lbp_kikuchi.utils.config import cfg_to_dict
from lbp_kikuchi.utils.logger import Logger
from lbp_kikuchi.utils.seed import seed_everything


def make_splits(N: int, val_frac: float, test_frac: float, seed: int) -> tuple:
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(N, generator=g).tolist()

    n_test = int(test_frac * N)
    n_val = int(val_frac * N)

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    if len(train_idx) == 0:
        raise ValueError(
            f"Train split is empty (N={N}, val_frac={val_frac}, test_frac={test_frac}). "
            "Reduce val/test fractions."
        )
    return train_idx, val_idx, test_idx


@hydra.main(version_base=None, config_path="../configs", config_name="pair")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}"
          + (f"  [{torch.cuda.get_device_name(0)}]" if device.type == "cuda" else ""))

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    logger = Logger(run_dir)
    logger.log_config(cfg_to_dict(cfg))
    writer = SummaryWriter(log_dir=run_dir / "tensorboard")

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path = Path(cfg.data.path).resolve()
    X = np.load(data_path / cfg.data.patterns_file)
    y_strain = np.load(data_path / cfg.data.strain_file)

    grid_rows = int(cfg.data.grid_rows)
    grid_cols = int(cfg.data.grid_cols)
    directions = tuple(OmegaConf.to_container(cfg.data.directions))

    print(f"Loaded  X       : {X.shape}  dtype={X.dtype}")
    print(f"Loaded  y       : {y_strain.shape}  dtype={y_strain.dtype}")
    print(f"Scan grid       : {grid_rows} × {grid_cols} = {grid_rows * grid_cols} points")
    print(f"Pair directions : {directions}")

    # ── Build pair index — index arrays only, no pattern data allocated ────────
    # Memory: O(M × 4 × int32) ≈ negligible vs O(M × 18 × H × W × float32)
    idx_a, idx_b, pos_a, pos_b = build_pair_index(
        grid_rows, grid_cols, directions=directions
    )
    M = len(idx_a)

    # Δε labels come directly from y_strain via the index arrays — no copies.
    delta_strain = y_strain[idx_b] - y_strain[idx_a]   # (M, 6)
    print(
        f"Pair samples    : {M}  "
        f"(Δε mean={delta_strain.mean(axis=0).round(6).tolist()}, "
        f"std={delta_strain.std(axis=0).round(6).tolist()})"
    )

    test_frac = float(getattr(cfg.training, "test_split", 0.0))
    train_idx, val_idx, test_idx = make_splits(
        M, float(cfg.training.val_split), test_frac, cfg.seed
    )
    print(f"Split   train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    with open(run_dir / "checkpoints" / "split_indices.json", "w") as f:
        json.dump({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}, f)

    # ── Target normalisation (Δε, train split only) ────────────────────────────
    dy_train = delta_strain[train_idx]
    y_mean = dy_train.mean(axis=0)
    y_std = dy_train.std(axis=0) + 1e-8
    delta_strain_norm = (delta_strain - y_mean) / y_std

    # ── Input normalisation ────────────────────────────────────────────────────
    # Compute stats on the raw flat X (no grid materialisation).  All patterns
    # originate from the same physical scan so including val/test patterns in the
    # stats computation is harmless; the max/min/mean/std are essentially the same
    # as computing on training patterns only, and avoids a large fancy-index copy.
    X_sq = X[:, 0] if X.ndim == 4 else X   # (N, H, W) view, no copy
    train_stats = compute_norm_stats(X_sq, cfg.training.norm_method)

    norm_stats_payload = {
        "norm_method": cfg.training.norm_method,
        **train_stats,
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }
    with open(run_dir / "checkpoints" / "norm_stats.json", "w") as f:
        json.dump(norm_stats_payload, f, indent=2)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    # LazyGridPairDataset extracts the two 3×3 patches per pair in __getitem__,
    # so the DataLoader never allocates more than (batch_size × 18 × H × W) at
    # once instead of the full (M × 18 × H × W) grid array.
    def make_ds(split_idx):
        return LazyGridPairDataset(
            X_sq,
            grid_rows, grid_cols,
            idx_a[split_idx],
            idx_b[split_idx],
            {"strain": delta_strain_norm[split_idx]},
            pos_a=pos_a[split_idx],
            pos_b=pos_b[split_idx],
            stats=train_stats,
            norm_method=cfg.training.norm_method,
        )

    loader_kw = dict(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(make_ds(train_idx), shuffle=True, **loader_kw)
    val_loader = DataLoader(make_ds(val_idx), shuffle=False, **loader_kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    img_size = int(getattr(cfg.model, "img_size", 224))
    model = PairModel(feature_dim=cfg.model.feature_dim, img_size=img_size).to(device)

    loss_fn     = str(cfg.training.loss_fn)
    huber_delta = float(cfg.training.huber_delta)
    sv_weight      = float(getattr(cfg.training, "sv_weight",      0.1))
    bounds_weight  = float(getattr(cfg.training, "bounds_weight",  0.01))
    max_abs_strain = float(getattr(cfg.training, "max_abs_strain", 0.05))

    logger.log_model(
        model,
        extra={
            "loss_fn": loss_fn,
            "huber_delta": huber_delta,
            "sv_weight": sv_weight,
            "bounds_weight": bounds_weight,
            "max_abs_strain": max_abs_strain,
            "feature_dim": cfg.model.feature_dim,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "directions": list(directions),
            "n_pairs": M,
        },
    )
    print(
        f"Model         : {type(model).__name__}  "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    print(
        f"Loss          : {loss_fn}"
        + (f"  delta={huber_delta}" if loss_fn == "huber" else "")
        + f"  sv_weight={sv_weight}  bounds_weight={bounds_weight}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(cfg.training.epochs):
        train_metrics = train_one_epoch_pair(
            model, train_loader, optimizer, device,
            loss_fn=loss_fn,
            huber_delta=huber_delta,
            sv_weight=sv_weight,
            bounds_weight=bounds_weight,
            max_abs_strain=max_abs_strain,
            epoch=epoch,
        )
        val_metrics = evaluate_pair(
            model, val_loader, device,
            loss_fn=loss_fn,
            huber_delta=huber_delta,
            sv_weight=sv_weight,
            bounds_weight=bounds_weight,
            max_abs_strain=max_abs_strain,
            epoch=epoch,
        )

        lr = optimizer.param_groups[0]["lr"]
        log = {
            "epoch": epoch,
            "lr": lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        print(log)
        logger.log(log)

        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Loss_SV/train", train_metrics["loss_sv"], epoch)
        writer.add_scalar("Loss_SV/val", val_metrics["loss_sv"], epoch)
        writer.add_scalar("Loss_Bounds/train", train_metrics["loss_bounds"], epoch)
        writer.add_scalar("Loss_Bounds/val", val_metrics["loss_bounds"], epoch)
        writer.add_scalar("DeltaStrainMAE/train", train_metrics["strain_mae"], epoch)
        writer.add_scalar("DeltaStrainMAE/val", val_metrics["strain_mae"], epoch)
        writer.add_scalar("DeltaStrainRMSE/train", train_metrics["strain_rmse"], epoch)
        writer.add_scalar("DeltaStrainRMSE/val", val_metrics["strain_rmse"], epoch)
        writer.add_scalar("LR", lr, epoch)
        for comp in ["e11", "e22", "e33", "e23", "e13", "e12"]:
            writer.add_scalar(
                f"PerComponentMAE_val/delta_{comp}", val_metrics[f"mae_{comp}"], epoch
            )

        scheduler.step()

        torch.save(model.state_dict(), run_dir / "checkpoints" / "last.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pt")

    writer.close()
    print(f"\nRun saved to: {run_dir}")


if __name__ == "__main__":
    main()
