"""
Single-pattern inference: load a trained encoder+strain-head and predict strain
for one (or a few) EBSD patterns.

Usage (from repo root):
    # Use the most-recent training run automatically:
    python scripts/infer.py

    # Point at a specific run:
    python scripts/infer.py --run-dir outputs/2026-04-28/16-14-37

    # Override default data path:
    python scripts/infer.py --data-dir data/ --index 5

    # Use last.pt instead of best.pt:
    python scripts/infer.py --checkpoint last.pt
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # safe for headless; change to "TkAgg" for interactive display
import matplotlib.pyplot as plt
import numpy as np
import torch

from lbp_kikuchi.data.dataset import apply_norm
from lbp_kikuchi.models.single_model import SinglePatternModel

VOIGT_LABELS = ["ε11", "ε22", "ε33", "ε23", "ε13", "ε12"]


# ── helpers ─────────────────────────────────────────────────────────────────

def find_latest_run(outputs_root: Path) -> Path:
    """Return the most recently modified run directory under outputs/."""
    candidates = sorted(
        (p for p in outputs_root.glob("*/*") if (p / "checkpoints").is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No completed run directories found under {outputs_root}. "
            "Run train_encoder.py first."
        )
    return candidates[-1]


def load_run(run_dir: Path, checkpoint_name: str = "best.pt"):
    """Load config, norm stats, and model weights from a run directory."""
    config_path = run_dir / "config_snapshot.json"
    norm_path = run_dir / "checkpoints" / "norm_stats.json"
    ckpt_path = run_dir / "checkpoints" / checkpoint_name

    for p in (config_path, norm_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    cfg = json.loads(config_path.read_text())
    norm_payload = json.loads(norm_path.read_text())

    norm_method = norm_payload.pop("norm_method")
    norm_stats = norm_payload  # remaining keys: min/max or mean/std

    return cfg, norm_method, norm_stats, ckpt_path


def preprocess(raw: np.ndarray, norm_stats: dict, norm_method: str) -> torch.Tensor:
    """
    raw: (H, W) or (1, H, W) single pattern — numpy float32/uint16/etc.
    Returns: (1, 1, H, W) float32 tensor ready for the model.
    """
    raw = raw.astype(np.float32)
    normed = apply_norm(raw, norm_stats, norm_method)
    # Squeeze out any existing channel dim then add batch+channel.
    if normed.ndim == 3 and normed.shape[0] == 1:
        normed = normed[0]           # (H, W)
    if normed.ndim == 2:
        normed = normed[np.newaxis, np.newaxis]  # (1, 1, H, W)
    return torch.from_numpy(normed)


def plot_inference(raw_pattern: np.ndarray, strain: np.ndarray,
                   index: int, out_path: Path) -> None:
    """
    Save a side-by-side figure: the Kikuchi pattern on the left,
    predicted Voigt strain as a horizontal bar chart on the right.
    """
    # Squeeze channel dim if present → (H, W)
    pat = raw_pattern.squeeze()

    fig, (ax_pat, ax_bar) = plt.subplots(1, 2, figsize=(12, 5),
                                          gridspec_kw={"width_ratios": [1.4, 1]})

    ax_pat.imshow(pat, cmap="gray", origin="lower")
    ax_pat.set_title(f"EBSD Pattern — sample #{index}", fontsize=11)
    ax_pat.axis("off")

    colors = ["steelblue" if v >= 0 else "tomato" for v in strain]
    bars = ax_bar.barh(VOIGT_LABELS[::-1], strain[::-1], color=colors[::-1],
                       edgecolor="black", linewidth=0.5)
    ax_bar.axvline(0, color="black", linewidth=0.8)
    ax_bar.set_xlabel("Predicted strain value", fontsize=10)
    ax_bar.set_title("Predicted Voigt Strain", fontsize=11)
    ax_bar.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, strain[::-1]):
        ax_bar.text(
            bar.get_width() + np.sign(val) * abs(strain).max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.5f}", va="center", ha="left" if val >= 0 else "right", fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to : {out_path}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Path to a specific Hydra run directory. "
                             "Default: auto-select the latest run under outputs/.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Directory containing X_patterns.npy (default: data/).")
    parser.add_argument("--patterns-file", default="X_patterns.npy",
                        help="Patterns numpy filename (default: X_patterns.npy).")
    parser.add_argument("--index", type=int, default=0,
                        help="Index of the sample to run inference on (default: 0).")
    parser.add_argument("--checkpoint", default="best.pt",
                        help="Checkpoint filename inside checkpoints/ (default: best.pt).")
    parser.add_argument("--save-plot", type=Path, default=None, metavar="FILE",
                        help="Save a pattern+prediction figure to FILE (e.g. pred.png). "
                             "Omit to skip plotting.")
    args = parser.parse_args()

    outputs_root = Path("outputs")
    run_dir = args.run_dir if args.run_dir else find_latest_run(outputs_root)
    print(f"Run directory : {run_dir}")

    cfg, norm_method, norm_stats, ckpt_path = load_run(run_dir, args.checkpoint)
    print(f"Checkpoint    : {ckpt_path}")
    print(f"Norm method   : {norm_method}  stats={norm_stats}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = SinglePatternModel(
        feature_dim=cfg["model"]["feature_dim"],
        predict_orientation=cfg["training"]["predict_orientation"],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    # ── Data ──────────────────────────────────────────────────────────────
    X = np.load(args.data_dir / args.patterns_file)
    print(f"Patterns shape: {X.shape}  dtype={X.dtype}")

    raw = X[args.index]
    x = preprocess(raw, norm_stats, norm_method).to(device)  # (1, 1, H, W)

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(x)

    strain = outputs["strain"].cpu().numpy()[0]

    print(f"\n{'='*40}")
    print(f"  Sample {args.index} — Predicted Voigt Strain")
    print(f"{'='*40}")
    for label, val in zip(VOIGT_LABELS, strain):
        bar = "▓" * int(abs(val) / (max(abs(strain)) + 1e-9) * 15 + 0.5)
        sign = "+" if val >= 0 else "-"
        print(f"  {label} : {val:+.6f}  {sign}{bar}")

    if "orientation" in outputs:
        q = outputs["orientation"].cpu().numpy()[0]
        print(f"\n  quaternion [q0, q1, q2, q3]:")
        print(f"    {q}")

    # ── Optional figure ───────────────────────────────────────────────────
    if args.save_plot:
        plot_inference(raw, strain, args.index, args.save_plot)


if __name__ == "__main__":
    main()
