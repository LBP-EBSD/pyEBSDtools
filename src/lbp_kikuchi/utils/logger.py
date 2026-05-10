import json
import csv
from pathlib import Path

import torch.nn as nn


class Logger:
    """
    Per-run experiment logger.

    On creation:
      - Saves a config snapshot to config_snapshot.json.

    Convenience methods:
      - log_config(cfg_dict)  — once at startup, writes config_snapshot.json
      - log_model(model, extra) — once at startup, writes model_summary.json
      - log(row_dict)         — per epoch, appends to metrics.json + metrics.csv

    Usage:
        logger = Logger(run_dir)
        logger.log_config(cfg_to_dict(cfg))
        logger.log_model(model, extra={"loss_fn": "huber", "huber_delta": 0.5})

        for epoch in range(epochs):
            ...
            logger.log({'epoch': epoch, 'train_loss': ..., 'val_mae': ...})
    """

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.json_file = self.out_dir / "metrics.json"
        self.csv_file = self.out_dir / "metrics.csv"

        self.logs: list[dict] = []
        self._csv_header_written = False

    def log_config(self, config: dict) -> None:
        """Persist config/hyperparameters used for this run."""
        with open(self.out_dir / "config_snapshot.json", "w") as f:
            json.dump(config, f, indent=2)

    def log_model(self, model: nn.Module, extra: dict | None = None) -> None:
        """
        Write model_summary.json with architecture info and parameter counts.

        Args:
            model: The PyTorch model.
            extra: Optional dict of additional key-value pairs to include
                   (e.g. loss_fn, huber_delta, predict_orientation).
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        per_module: dict[str, int] = {}
        for name, module in model.named_children():
            n = sum(p.numel() for p in module.parameters())
            per_module[name] = n

        summary = {
            "class": type(model).__name__,
            "n_params_total": total_params,
            "n_params_trainable": trainable_params,
            "per_module_params": per_module,
        }
        if extra:
            summary.update(extra)

        with open(self.out_dir / "model_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def log(self, data: dict) -> None:
        """Append one row of metrics (e.g. one epoch)."""
        self.logs.append(data)

        # Full JSON history (rewritten each time so it's always valid JSON).
        with open(self.json_file, "w") as f:
            json.dump(self.logs, f, indent=2)

        # CSV: write header once, then append rows.
        if not self._csv_header_written:
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(data.keys()))
                writer.writeheader()
                writer.writerow(data)
            self._csv_header_written = True
        else:
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(data.keys()))
                writer.writerow(data)
