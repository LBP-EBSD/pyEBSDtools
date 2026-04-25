# src/lbp_kikuchi/utils/logger.py

import json
from pathlib import Path

class Logger:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.out_dir / "metrics.json"
        self.logs = []

    def log(self, data):
        self.logs.append(data)
        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=2)