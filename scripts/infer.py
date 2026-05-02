"""
Single-pattern inference: load a trained model and predict strain for one sample.

Usage (from repo root):
    # Use the most-recent run, sample index 0:
    python scripts/infer.py

    # Specific run and sample:
    python scripts/infer.py --run-dir outputs/2026-04-29/14-00-58 --index 5

    # Save a pattern + bar-chart figure:
    python scripts/infer.py --index 3 --save-plot pred.png

    # Override data directory (default: taken from the run's config):
    python scripts/infer.py --data-dir data/custom/

    # Use last.pt instead of best.pt:
    python scripts/infer.py --checkpoint last.pt
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

from lbp_kikuchi.data.dataset import apply_norm
from lbp_kikuchi.models.single_model import SinglePatternModel

VOIGT_LABELS = ["ε11", "ε22", "ε33", "ε23", "ε13", "ε12"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_latest_run(outputs_root: Path) -> Path:
    candidates = sorted(
        (p for p in outputs_root.glob("*/*") if (p / "checkpoints").is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No completed runs found under {outputs_root}. "
            "Run train_encoder.py first."
        )
    return candidates[-1]


def load_run(run_dir: Path, checkpoint_name: str = "best.pt") -> tuple:
    config_path = run_dir / "config_snapshot.json"
    norm_path   = run_dir / "checkpoints" / "norm_stats.json"
    ckpt_path   = run_dir / "checkpoints" / checkpoint_name

    for p in (config_path, norm_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    cfg = json.loads(config_path.read_text())

    norm_payload = json.loads(norm_path.read_text())
    norm_method = norm_payload.pop("norm_method")
    norm_stats = norm_payload

    missing = [k for k in ("y_mean", "y_std") if k not in norm_stats]
    if missing:
        raise KeyError(
            f"Missing target normalisation keys {missing} in {norm_path}. "
            "Re-train with the current train_encoder.py."
        )

    return cfg, norm_method, norm_stats, ckpt_path


def preprocess(
    raw: np.ndarray, norm_stats: dict, norm_method: str
) -> torch.Tensor:
    """
    raw: (H, W) or (1, H, W) single pattern.
    Returns: (1, 1, H, W) float32 tensor.
    """
    raw = raw.astype(np.float32)
    normed = apply_norm(raw, norm_stats, norm_method)
    if normed.ndim == 3 and normed.shape[0] == 1:
        normed = normed[0]
    if normed.ndim == 2:
        normed = normed[np.newaxis, np.newaxis]   # (1, 1, H, W)
    return torch.from_numpy(normed)


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_inference(
    raw_pattern: np.ndarray,
    strain: np.ndarray,
    index: int,
    out_path: Path,
) -> None:
    pat = raw_pattern.squeeze()

    fig, (ax_pat, ax_bar) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.4, 1]}
    )

    ax_pat.imshow(pat, cmap="gray", origin="lower")
    ax_pat.set_title(f"EBSD Pattern — sample #{index}", fontsize=11)
    ax_pat.axis("off")

    colors = ["steelblue" if v >= 0 else "tomato" for v in strain]
    bars = ax_bar.barh(
        VOIGT_LABELS[::-1], strain[::-1],
        color=colors[::-1], edgecolor="black", linewidth=0.5,
    )
    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_xlabel("Predicted strain", fontsize=10)
    ax_bar.set_title("Predicted Voigt Strain", fontsize=11)
    ax_bar.grid(axis="x", alpha=0.3)

    scale = abs(strain).max() or 1e-9
    for bar, val in zip(bars, strain[::-1]):
        ax_bar.text(
            bar.get_width() + np.sign(val) * scale * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.5f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Run directory. Default: latest under outputs/.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Data directory. Default: taken from the run's config_snapshot.json.",
    )
    parser.add_argument("--patterns-file", default="X_patterns.npy")
    parser.add_argument(
        "--index", type=int, default=0,
        help="Sample index to run inference on. Default: 0.",
    )
    parser.add_argument(
        "--checkpoint", default="best.pt",
        help="Checkpoint filename inside checkpoints/. Default: best.pt.",
    )
    parser.add_argument(
        "--save-plot", type=Path, default=None, metavar="FILE",
        help="Save a pattern+prediction figure to FILE (e.g. pred.png).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run(Path("outputs"))
    print(f"Run dir     : {run_dir}")

    cfg, norm_method, norm_stats, ckpt_path = load_run(run_dir, args.checkpoint)
    print(f"Checkpoint  : {ckpt_path}")
    print(f"Norm method : {norm_method}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

    model = SinglePatternModel(
        feature_dim=cfg["model"]["feature_dim"],
        predict_orientation=cfg["training"]["predict_orientation"],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()

    data_dir = args.data_dir or Path(cfg["data"]["path"])
    X = np.load(data_dir / args.patterns_file)
    print(f"Patterns    : {X.shape}")

    raw = X[args.index]
    x = preprocess(raw, norm_stats, norm_method).to(device)

    with torch.no_grad():
        outputs = model(x)

    strain = outputs["strain"].cpu().numpy()[0]
    # Denormalise back to physical strain units.
    y_mean = np.array(norm_stats["y_mean"])
    y_std  = np.array(norm_stats["y_std"])
    strain = strain * y_std + y_mean

    print(f"\n{'='*42}")
    print(f"  Sample {args.index} — Predicted Voigt Strain")
    print(f"{'='*42}")
    scale = abs(strain).max() or 1e-9
    for label, val in zip(VOIGT_LABELS, strain):
        bar = "▓" * int(abs(val) / scale * 15 + 0.5)
        print(f"  {label} : {val:+.6f}  {bar}")

    if "orientation" in outputs:
        q = outputs["orientation"].cpu().numpy()[0]
        print(f"\n  Quaternion [q0, q1, q2, q3]: {q}")

    if args.save_plot:
        plot_inference(raw, strain, args.index, args.save_plot)


if __name__ == "__main__":
    main()
