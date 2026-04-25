import json
import csv
from pathlib import Path


class Logger:
    """
    Per-run experiment logger.

    On creation:
      - Saves a config snapshot to config_snapshot.json.

    Each call to log():
      - Appends to metrics.json (full history, human-readable).
      - Appends a row to metrics.csv  (easy to load with pandas/numpy).

    Usage:
        logger = Logger(run_dir)
        logger.log_config(cfg_to_dict(cfg))   # once at startup

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
