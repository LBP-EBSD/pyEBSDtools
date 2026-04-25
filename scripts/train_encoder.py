import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path

from lbp_kikuchi.data.dataset import EBSDDataset
from lbp_kikuchi.models.encoder import Encoder
from lbp_kikuchi.models.heads import StrainHead
from lbp_kikuchi.training.engine import train_one_epoch, evaluate
from lbp_kikuchi.utils.logger import Logger


def load_data(path):
    X = np.load(f"{path}/X_patterns.npy")
    y = np.load(f"{path}/y_strain.npy")
    return X, y


@hydra.main(version_base=None, config_path="../configs", config_name="encoder")
def main(cfg: DictConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hydra changes working dir → use absolute path
    data_path = Path(cfg.data.path).resolve()

    X, y = load_data(data_path)
    dataset = EBSDDataset(X, y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size
    )

    encoder = Encoder(cfg.model.feature_dim)
    head = StrainHead(cfg.model.feature_dim)

    model = torch.nn.Sequential(encoder, head).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    # Hydra run dir (auto-created per run)
    run_dir = Path.cwd()
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    logger = Logger(run_dir)

    for epoch in range(cfg.training.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        }

        print(log)
        logger.log(log)

        torch.save(model.state_dict(), run_dir / f"checkpoints/epoch_{epoch}.pt")


if __name__ == "__main__":
    main()