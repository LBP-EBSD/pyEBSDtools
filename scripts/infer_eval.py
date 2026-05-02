"""
Full evaluation of a trained encoder+strain-head.

Usage (from repo root):
    # Evaluate val split of the most-recent run (default):
    python scripts/infer_eval.py

    # Evaluate a specific run:
    python scripts/infer_eval.py --run-dir outputs/2026-04-29/14-00-58

    # Choose which split to evaluate:
    python scripts/infer_eval.py --split val      # default: held-out val set
    python scripts/infer_eval.py --split train    # training set  (sanity/overfit check)
    python scripts/infer_eval.py --split test     # held-out test set (if test_split > 0)
    python scripts/infer_eval.py --split all      # entire dataset

    # Override data directory (default: taken from run's config_snapshot.json):
    python scripts/infer_eval.py --data-dir /path/to/data

    # Other options:
    python scripts/infer_eval.py --checkpoint last.pt
    python scripts/infer_eval.py --batch-size 128
    python scripts/infer_eval.py --max-samples 1000
    python scripts/infer_eval.py --spot-checks 10
    python scripts/infer_eval.py --no-plot

Outputs (written to the run directory):
    eval_scatter_<split>.png
    eval_results_<split>.json
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


# ── Run helpers ──────────────────────────────────────────────────────────────

def find_latest_run(outputs_root: Path) -> Path:
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


def load_run(run_dir: Path, checkpoint_name: str = "best.pt") -> tuple:
    """
    Load everything needed for inference from a run directory.

    Returns: (cfg, norm_method, norm_stats, ckpt_path)
        norm_stats contains both input stats (min/max or mean/std)
        and target stats (y_mean, y_std).
    """
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
            "This checkpoint was produced by an older pipeline. Re-train with the "
            "current train_encoder.py to use this evaluator."
        )

    return cfg, norm_method, norm_stats, ckpt_path


def select_split_indices(run_dir: Path, split: str, n_total: int) -> np.ndarray:
    """
    Return the index array for the requested split.

    split ∈ {"train", "val", "test", "all"}
    """
    if split == "all":
        return np.arange(n_total, dtype=np.int64)

    split_path = run_dir / "checkpoints" / "split_indices.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split index file not found: {split_path}. "
            "Re-train with the current train_encoder.py to save split indices, "
            "or use --split all."
        )

    payload = json.loads(split_path.read_text())
    key = f"{split}_idx"
    if key not in payload:
        raise KeyError(
            f"Key '{key}' not found in {split_path}. "
            f"Available keys: {list(payload.keys())}"
        )

    idx = np.array(payload[key], dtype=np.int64)
    if idx.size == 0:
        raise ValueError(
            f"Split '{split}' is empty. "
            "Check training config (test_split / val_split fractions)."
        )
    if idx.min() < 0 or idx.max() >= n_total:
        raise ValueError(
            f"Split '{split}' contains out-of-range indices for dataset size {n_total}."
        )
    return idx


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_batch(
    raw: np.ndarray, norm_stats: dict, norm_method: str
) -> torch.Tensor:
    """
    raw: (B, H, W) or (B, 1, H, W) float32 array.
    Returns: (B, 1, H, W) float32 tensor.
    """
    normed = apply_norm(raw.astype(np.float32), norm_stats, norm_method)
    if normed.ndim == 3:
        normed = normed[:, np.newaxis]
    return torch.from_numpy(normed)


# ── Evaluation ───────────────────────────────────────────────────────────────

def run_eval(
    model: torch.nn.Module,
    X: np.ndarray,
    y_strain: np.ndarray,
    norm_stats: dict,
    norm_method: str,
    device: torch.device,
    batch_size: int = 64,
    max_samples: int | None = None,
) -> dict:
    N = len(X) if max_samples is None else min(max_samples, len(X))
    X, y_strain = X[:N], y_strain[:N]

    all_preds: list[np.ndarray] = []
    model.eval()

    for start in range(0, N, batch_size):
        xb = preprocess_batch(X[start : start + batch_size], norm_stats, norm_method)
        with torch.no_grad():
            preds = model(xb.to(device))["strain"].cpu().numpy()
        all_preds.append(preds)

    preds = np.concatenate(all_preds, axis=0)   # (N, 6)

    # Denormalise back to physical strain units.
    y_mean = np.array(norm_stats["y_mean"])
    y_std  = np.array(norm_stats["y_std"])
    preds  = preds * y_std + y_mean

    errors = np.abs(preds - y_strain)
    per_component_mae = errors.mean(axis=0)

    return {
        "n_samples": N,
        "overall_mae": float(errors.mean()),
        "rmse": float(np.sqrt(((preds - y_strain) ** 2).mean())),
        "max_abs_error": float(errors.max()),
        "per_component_mae": {
            lbl: float(v) for lbl, v in zip(VOIGT_LABELS, per_component_mae)
        },
        "preds": preds,
        "targets": y_strain,
    }


# ── Console output ───────────────────────────────────────────────────────────

def print_results(results: dict, split: str) -> None:
    N = results["n_samples"]
    comp_maes = results["per_component_mae"]
    max_mae = max(comp_maes.values()) or 1e-9

    print(f"\n{'='*58}")
    print(f"  Evaluation — split={split}  N={N}")
    print(f"{'='*58}")
    print(f"  {'Metric':<18} {'Value':>12}")
    print(f"  {'-'*32}")
    print(f"  {'Overall MAE':<18} {results['overall_mae']:>12.6f}")
    print(f"  {'RMSE':<18} {results['rmse']:>12.6f}")
    print(f"  {'Max |error|':<18} {results['max_abs_error']:>12.6f}")
    print(f"\n  Per-component MAE (Voigt order):")
    print(f"  {'Component':<10} {'MAE':>10}  bar")
    print(f"  {'-'*44}")
    for lbl, mae in comp_maes.items():
        bar = "█" * int(mae / max_mae * 20 + 0.5)
        print(f"  {lbl:<10} {mae:>10.6f}  {bar}")
    print(f"{'='*58}\n")


def print_spot_checks(
    X: np.ndarray,
    y_strain: np.ndarray,
    model: torch.nn.Module,
    norm_stats: dict,
    norm_method: str,
    device: torch.device,
    n: int = 5,
) -> None:
    y_mean = np.array(norm_stats["y_mean"])
    y_std  = np.array(norm_stats["y_std"])
    print(f"\n  Spot-check (first {n} samples):")
    print(f"  {'idx':>4}  {'mean |err|':>11}  {'pred (ε11..ε12)':>46}  true")
    for i in range(min(n, len(X))):
        xb = preprocess_batch(X[i : i + 1], norm_stats, norm_method).to(device)
        with torch.no_grad():
            pred = model(xb)["strain"].cpu().numpy()[0]
        pred = pred * y_std + y_mean
        gt = y_strain[i]
        mae_i = float(np.abs(pred - gt).mean())
        pred_str = " ".join(f"{v:+.4f}" for v in pred)
        true_str = " ".join(f"{v:+.4f}" for v in gt)
        print(f"  {i:>4}  {mae_i:>11.6f}  {pred_str}  {true_str}")


# ── Scatter plot ─────────────────────────────────────────────────────────────

def plot_scatter(results: dict, out_path: Path, split: str) -> None:
    preds   = results["preds"]
    targets = results["targets"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, (ax, lbl) in enumerate(zip(axes, VOIGT_LABELS)):
        p = preds[:, i]
        t = targets[:, i]
        mae = float(np.abs(p - t).mean())

        if len(p) > 2000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(p), 2000, replace=False)
            p, t = p[idx], t[idx]

        lo = min(p.min(), t.min())
        hi = max(p.max(), t.max())
        pad = (hi - lo) * 0.05 or 1e-4

        ax.scatter(t, p, s=6, alpha=0.4, linewidths=0, color="steelblue")
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.2, label="ideal")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("True", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(f"{lbl}  (MAE={mae:.4f})", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Predicted vs True — Voigt Strain   [{split} split]\n"
        f"N={results['n_samples']}  Overall MAE={results['overall_mae']:.4f}"
        f"  RMSE={results['rmse']:.4f}",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot → {out_path}")


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
        "--split", choices=("val", "train", "test", "all"), default="val",
        help="Which split to evaluate. Default: val.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Data directory. Default: taken from the run's config_snapshot.json.",
    )
    parser.add_argument("--patterns-file", default="X_patterns.npy")
    parser.add_argument("--strain-file", default="y_strain.npy")
    parser.add_argument(
        "--checkpoint", default="best.pt",
        help="Checkpoint filename inside checkpoints/. Default: best.pt.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap evaluation at N samples (default: all in the split).",
    )
    parser.add_argument(
        "--spot-checks", type=int, default=5,
        help="Number of individual samples to print. Default: 5.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip scatter plot.")
    args = parser.parse_args()

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir = args.run_dir or find_latest_run(Path("outputs"))
    print(f"Run dir       : {run_dir}")

    cfg, norm_method, norm_stats, ckpt_path = load_run(run_dir, args.checkpoint)
    print(f"Checkpoint    : {ckpt_path}")
    print(f"Norm method   : {norm_method}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device        : {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SinglePatternModel(
        feature_dim=cfg["model"]["feature_dim"],
        predict_orientation=cfg["training"]["predict_orientation"],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device).eval()

    # ── Data ──────────────────────────────────────────────────────────────────
    # Prefer explicit --data-dir; fall back to path recorded in the run config.
    data_dir = args.data_dir or Path(cfg["data"]["path"])
    print(f"Data dir      : {data_dir}")

    X = np.load(data_dir / args.patterns_file)
    y_strain = np.load(data_dir / args.strain_file)
    print(f"Patterns      : {X.shape}  dtype={X.dtype}")
    print(f"Labels        : {y_strain.shape}  dtype={y_strain.dtype}")
    assert len(X) == len(y_strain), "Patterns and labels must have the same length."

    # ── Split selection ───────────────────────────────────────────────────────
    split_idx = select_split_indices(run_dir, args.split, len(X))
    X_eval = X[split_idx]
    y_eval = y_strain[split_idx]
    print(f"Eval split    : {args.split}  n={len(X_eval)}")

    # ── Spot checks ───────────────────────────────────────────────────────────
    print_spot_checks(
        X_eval, y_eval, model, norm_stats, norm_method, device, n=args.spot_checks
    )

    # ── Full evaluation ───────────────────────────────────────────────────────
    results = run_eval(
        model, X_eval, y_eval, norm_stats, norm_method, device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    print_results(results, args.split)

    # ── Scatter plot ──────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_scatter(results, run_dir / f"eval_scatter_{args.split}.png", args.split)

    # ── Save metrics ──────────────────────────────────────────────────────────
    out_path = run_dir / f"eval_results_{args.split}.json"
    saveable = {k: v for k, v in results.items() if k not in ("preds", "targets")}
    out_path.write_text(json.dumps(saveable, indent=2))
    print(f"Results       → {out_path}")


if __name__ == "__main__":
    main()
