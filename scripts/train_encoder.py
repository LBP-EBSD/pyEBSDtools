"""
Phase 1 training: single Kikuchi pattern → strain (+ optional orientation).

Run from repo root:
    python scripts/train_encoder.py
    python scripts/train_encoder.py training.lr=5e-4 model.feature_dim=256
    python scripts/train_encoder.py training.predict_orientation=true
    python scripts/train_encoder.py data.path=data/overfit64 training.epochs=200

Config overrides (Hydra syntax, no -- prefix):
    training.epochs=100
    training.batch_size=64
    training.lr=5e-4
    training.val_split=0.15
    training.test_split=0.05      # set 0.0 to disable test split
    training.norm_method=zscore   # or minmax
    model.feature_dim=256
    data.path=data/custom/
    experiment_name=my_run

Outputs (under outputs/YYYY-MM-DD/HH-MM-SS/):
    checkpoints/best.pt            lowest val loss checkpoint
    checkpoints/last.pt            end-of-training checkpoint
    checkpoints/norm_stats.json    input + target normalisation stats
    checkpoints/split_indices.json train/val/test index arrays
    config_snapshot.json           exact config used
    metrics.csv / metrics.json     per-epoch metrics
    tensorboard/                   TensorBoard event files
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

# Always import from this repo's src/, not any other installed copy.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from lbp_kikuchi.data.dataset import EBSDDataset, compute_norm_stats
from lbp_kikuchi.models.single_model import SinglePatternModel
from lbp_kikuchi.training.engine import train_one_epoch, evaluate
from lbp_kikuchi.utils.config import cfg_to_dict
from lbp_kikuchi.utils.logger import Logger
from lbp_kikuchi.utils.seed import seed_everything


def make_splits(N: int, val_frac: float, test_frac: float, seed: int) -> tuple:
    """
    Randomly partition N indices into (train, val, test).

    Order in the shuffled array: [test | val | train]
    so train is the largest contiguous tail, which is stable under re-runs
    with the same seed regardless of val/test fraction tweaks.

    Returns three lists of int indices.
    """
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


@hydra.main(version_base=None, config_path="../configs", config_name="encoder")
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

    print(f"Loaded  X     : {X.shape}  dtype={X.dtype}")
    print(f"Loaded  y     : {y_strain.shape}  dtype={y_strain.dtype}")

    N = len(X)
    test_frac = float(getattr(cfg.training, "test_split", 0.0))
    train_idx, val_idx, test_idx = make_splits(
        N, float(cfg.training.val_split), test_frac, cfg.seed
    )
    print(
        f"Split   train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}"
    )

    # Persist split indices so infer_eval.py can reproduce exact splits.
    split_payload = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    with open(run_dir / "checkpoints" / "split_indices.json", "w") as f:
        json.dump(split_payload, f)

    # ── Target normalisation (train split only — no leakage) ──────────────────
    y_train = y_strain[train_idx]
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0) + 1e-8
    y_strain_norm = (y_strain - y_mean) / y_std

    targets: dict = {"strain": y_strain_norm}
    if cfg.training.predict_orientation:
        targets["orientation"] = np.load(data_path / cfg.data.orientation_file)

    # ── Input normalisation (train split only) ─────────────────────────────────
    train_stats = compute_norm_stats(X[train_idx], cfg.training.norm_method)

    # Persist all normalisation stats for reproducible inference.
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
        return EBSDDataset(
            X[split_idx],
            {k: v[split_idx] for k, v in targets.items()},
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
    model = SinglePatternModel(
        feature_dim=cfg.model.feature_dim,
        predict_orientation=cfg.training.predict_orientation,
        img_size=img_size,
    ).to(device)

    loss_fn = str(cfg.training.loss_fn)
    huber_delta = float(cfg.training.huber_delta)
    logger.log_model(
        model,
        extra={
            "loss_fn": loss_fn,
            "huber_delta": huber_delta,
            "feature_dim": cfg.model.feature_dim,
            "predict_orientation": cfg.training.predict_orientation,
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
    best_val_loss = float("inf")

    for epoch in range(cfg.training.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            orientation_loss_weight=cfg.training.orientation_loss_weight,
            loss_fn=loss_fn,
            huber_delta=huber_delta,
        )
        val_metrics = evaluate(
            model, val_loader, device,
            orientation_loss_weight=cfg.training.orientation_loss_weight,
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
