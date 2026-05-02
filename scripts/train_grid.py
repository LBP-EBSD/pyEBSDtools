"""
Stage 2 training: 3×3 pattern grid → ε(center).

The scan is assumed to be stored as a flat (N, H, W) array where pattern i
lives at scan position (i // grid_cols, i % grid_cols) — standard row-major
order from EMsoft.  Supply grid_rows and grid_cols so the script can extract
3×3 neighbourhoods.  Only interior points (1 ≤ r ≤ rows-2, 1 ≤ c ≤ cols-2)
produce valid grids; boundary patterns are dropped.

NOTE on train/val leakage: adjacent grid samples share up to 6 of their 9
patterns, so a random split at the sample level will have pattern overlap
between splits.  This is acceptable for a development baseline.  For a
rigorous evaluation, split by scan region (pass disjoint rect crops as
separate data.path directories).

Run from repo root:
    python scripts/train_grid.py
    python scripts/train_grid.py data.grid_rows=100 data.grid_cols=100
    python scripts/train_grid.py training.lr=5e-4 model.feature_dim=256

Config overrides (Hydra syntax):
    data.grid_rows=100
    data.grid_cols=100
    training.epochs=50
    training.batch_size=16
    training.lr=1e-3
    model.feature_dim=128
    experiment_name=stage2_run1

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
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from lbp_kikuchi.data.dataset import GridDataset, build_grid_samples, compute_norm_stats
from lbp_kikuchi.models.grid_model import GridModel
from lbp_kikuchi.training.engine import evaluate, train_one_epoch
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


@hydra.main(version_base=None, config_path="../configs", config_name="grid")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print(f"Loaded  X     : {X.shape}  dtype={X.dtype}")
    print(f"Loaded  y     : {y_strain.shape}  dtype={y_strain.dtype}")
    print(f"Scan grid     : {grid_rows} × {grid_cols} = {grid_rows * grid_cols} points")

    # Build 3×3 grid samples from the flat scan data.
    grids, y_grids = build_grid_samples(X, y_strain, grid_rows, grid_cols)
    M = len(grids)
    print(f"Grid samples  : {M}  (interior points of {grid_rows}×{grid_cols} scan)")

    test_frac = float(getattr(cfg.training, "test_split", 0.0))
    train_idx, val_idx, test_idx = make_splits(
        M, float(cfg.training.val_split), test_frac, cfg.seed
    )
    print(f"Split   train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    with open(run_dir / "checkpoints" / "split_indices.json", "w") as f:
        json.dump({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}, f)

    # ── Target normalisation (train split only) ────────────────────────────────
    y_train = y_grids[train_idx]
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0) + 1e-8
    y_grids_norm = (y_grids - y_mean) / y_std

    # ── Input normalisation (train split only) ─────────────────────────────────
    # Stats computed over all patterns in all training grids.
    flat_train = grids[train_idx].reshape(-1, *X.shape[1:])
    train_stats = compute_norm_stats(flat_train, cfg.training.norm_method)

    norm_stats_payload = {
        "norm_method": cfg.training.norm_method,
        **train_stats,
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }
    with open(run_dir / "checkpoints" / "norm_stats.json", "w") as f:
        json.dump(norm_stats_payload, f, indent=2)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    def make_ds(split_idx):
        return GridDataset(
            grids[split_idx],
            {"strain": y_grids_norm[split_idx]},
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
    model = GridModel(feature_dim=cfg.model.feature_dim).to(device)

    loss_fn = str(cfg.training.loss_fn)
    huber_delta = float(cfg.training.huber_delta)
    logger.log_model(
        model,
        extra={
            "loss_fn": loss_fn,
            "huber_delta": huber_delta,
            "feature_dim": cfg.model.feature_dim,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "n_grid_samples": M,
        },
    )
    print(
        f"Model         : {type(model).__name__}  "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"Loss          : {loss_fn}"
          + (f"  delta={huber_delta}" if loss_fn == "huber" else ""))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    # GridDataset returns (grid, targets_dict) — same interface as EBSDDataset —
    # so train_one_epoch / evaluate work unchanged.
    best_val_loss = float("inf")

    for epoch in range(cfg.training.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            loss_fn=loss_fn,
            huber_delta=huber_delta,
        )
        val_metrics = evaluate(
            model, val_loader, device,
            loss_fn=loss_fn,
            huber_delta=huber_delta,
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
        writer.add_scalar("StrainMAE/train", train_metrics["strain_mae"], epoch)
        writer.add_scalar("StrainMAE/val", val_metrics["strain_mae"], epoch)
        writer.add_scalar("StrainRMSE/train", train_metrics["strain_rmse"], epoch)
        writer.add_scalar("StrainRMSE/val", val_metrics["strain_rmse"], epoch)
        writer.add_scalar("LR", lr, epoch)
        for comp in ["e11", "e22", "e33", "e23", "e13", "e12"]:
            writer.add_scalar(
                f"PerComponentMAE_val/{comp}", val_metrics[f"mae_{comp}"], epoch
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
