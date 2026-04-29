"""
Phase 1 training: single Kikuchi pattern → strain (+ orientation).

Run from repo root:
    python scripts/train_encoder.py
    python scripts/train_encoder.py training.lr=5e-4 model.feature_dim=256
    python scripts/train_encoder.py training.predict_orientation=true experiment_name=with_ori

Outputs (Hydra-managed, in outputs/YYYY-MM-DD/HH-MM-SS/):
    checkpoints/best.pt     lowest val loss
    checkpoints/last.pt     most recent epoch
    metrics.json / .csv     per-epoch metrics
    config_snapshot.json    exact config used
"""

import hydra
import json
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from lbp_kikuchi.data.dataset import EBSDDataset, compute_norm_stats
from lbp_kikuchi.models.single_model import SinglePatternModel
from lbp_kikuchi.training.engine import train_one_epoch, evaluate
from lbp_kikuchi.utils.config import cfg_to_dict
from lbp_kikuchi.utils.logger import Logger
from lbp_kikuchi.utils.seed import seed_everything


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

    targets = {"strain": np.load(data_path / cfg.data.strain_file)}
    if cfg.training.predict_orientation:
        targets["orientation"] = np.load(data_path / cfg.data.orientation_file)

    N = len(X)
    val_size = int(cfg.training.val_split * N)
    idx = torch.randperm(N).tolist()
    train_idx, val_idx = idx[val_size:], idx[:val_size]

    # Stats computed on train split only — applying train stats to val prevents leakage.
    train_stats = compute_norm_stats(X[train_idx], cfg.training.norm_method)

    # Persist norm stats so inference can reproduce identical normalisation.
    norm_stats_payload = {"norm_method": cfg.training.norm_method, **train_stats}
    with open(run_dir / "checkpoints" / "norm_stats.json", "w") as f:
        json.dump(norm_stats_payload, f, indent=2)

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
        pin_memory=True,
    )
    train_loader = DataLoader(make_ds(train_idx), shuffle=True, **loader_kw)
    val_loader = DataLoader(make_ds(val_idx), shuffle=False, **loader_kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SinglePatternModel(
        feature_dim=cfg.model.feature_dim,
        predict_orientation=cfg.training.predict_orientation,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    # ── Loop ──────────────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(cfg.training.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            orientation_loss_weight=cfg.training.orientation_loss_weight,
        )
        val_metrics = evaluate(
            model, val_loader, device,
            orientation_loss_weight=cfg.training.orientation_loss_weight,
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

        # ── TensorBoard ───────────────────────────────────────────────────
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("StrainMAE/train", train_metrics["strain_mae"], epoch)
        writer.add_scalar("StrainMAE/val", val_metrics["strain_mae"], epoch)
        writer.add_scalar("StrainRMSE/train", train_metrics["strain_rmse"], epoch)
        writer.add_scalar("StrainRMSE/val", val_metrics["strain_rmse"], epoch)
        writer.add_scalar("LR", lr, epoch)
        for comp in ["e11", "e22", "e33", "e23", "e13", "e12"]:
            writer.add_scalar(f"PerComponentMAE_val/{comp}", val_metrics[f"mae_{comp}"], epoch)

        scheduler.step()

        torch.save(model.state_dict(), run_dir / "checkpoints" / "last.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pt")

    writer.close()


if __name__ == "__main__":
    main()
