"""
Full evaluation of a trained encoder+strain-head on the labelled dataset.

Usage (from repo root):
    # Evaluate the most-recent run on the full dataset:
    python scripts/infer_eval.py

    # Point at a specific run:
    python scripts/infer_eval.py --run-dir outputs/2026-04-28/16-14-37

    # Evaluate on a subset of samples:
    python scripts/infer_eval.py --max-samples 500

    # Use last.pt instead of best.pt:
    python scripts/infer_eval.py --checkpoint last.pt

    # Larger batch for faster GPU evaluation:
    python scripts/infer_eval.py --batch-size 128

Output:
    Console: per-component MAE, overall MAE, RMSE, max error.
    File:    <run_dir>/eval_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch

from lbp_kikuchi.data.dataset import apply_norm
from lbp_kikuchi.models.single_model import SinglePatternModel

VOIGT_LABELS = ["ε11", "ε22", "ε33", "ε23", "ε13", "ε12"]


# ── helpers (shared with infer.py) ──────────────────────────────────────────

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


def load_run(run_dir: Path, checkpoint_name: str = "best.pt"):
    config_path = run_dir / "config_snapshot.json"
    norm_path = run_dir / "checkpoints" / "norm_stats.json"
    ckpt_path = run_dir / "checkpoints" / checkpoint_name

    for p in (config_path, norm_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    cfg = json.loads(config_path.read_text())
    norm_payload = json.loads(norm_path.read_text())
    norm_method = norm_payload.pop("norm_method")
    norm_stats = norm_payload

    return cfg, norm_method, norm_stats, ckpt_path


def preprocess_batch(raw: np.ndarray, norm_stats: dict, norm_method: str) -> torch.Tensor:
    """
    raw: (B, H, W) or (B, 1, H, W) numpy array.
    Returns: (B, 1, H, W) float32 tensor.
    """
    normed = apply_norm(raw.astype(np.float32), norm_stats, norm_method)
    if normed.ndim == 3:
        normed = normed[:, np.newaxis]   # (B, H, W) → (B, 1, H, W)
    # ndim == 4 means channel already present — pass through unchanged.
    return torch.from_numpy(normed)


# ── evaluation ──────────────────────────────────────────────────────────────

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
    """
    Returns a dict with scalar and per-component metrics.
    """
    N = len(X) if max_samples is None else min(max_samples, len(X))
    X, y_strain = X[:N], y_strain[:N]

    all_preds = []
    model.eval()

    for start in range(0, N, batch_size):
        xb = preprocess_batch(X[start : start + batch_size], norm_stats, norm_method)
        xb = xb.to(device)
        with torch.no_grad():
            preds = model(xb)["strain"].cpu().numpy()
        all_preds.append(preds)

    preds = np.concatenate(all_preds, axis=0)   # (N, 6)
    errors = np.abs(preds - y_strain)            # (N, 6)

    per_component_mae = errors.mean(axis=0)      # (6,)
    overall_mae = float(errors.mean())
    rmse = float(np.sqrt(((preds - y_strain) ** 2).mean()))
    max_err = float(errors.max())

    return {
        "n_samples": N,
        "overall_mae": overall_mae,
        "rmse": rmse,
        "max_abs_error": max_err,
        "per_component_mae": {lbl: float(v) for lbl, v in zip(VOIGT_LABELS, per_component_mae)},
        "preds": preds,
        "targets": y_strain,
    }


def print_results(results: dict) -> None:
    N = results["n_samples"]
    comp_maes = results["per_component_mae"]
    max_mae = max(comp_maes.values())

    print(f"\n{'='*56}")
    print(f"  Evaluation over {N} samples")
    print(f"{'='*56}")
    print(f"  {'Metric':<18} {'Value':>12}")
    print(f"  {'-'*30}")
    print(f"  {'Overall MAE':<18} {results['overall_mae']:>12.6f}")
    print(f"  {'RMSE':<18} {results['rmse']:>12.6f}")
    print(f"  {'Max |error|':<18} {results['max_abs_error']:>12.6f}")
    print(f"\n  Per-component MAE (Voigt order):")
    print(f"  {'Component':<10} {'MAE':>10}  {'':20}")
    print(f"  {'-'*42}")
    for lbl, mae in comp_maes.items():
        bar = "█" * int(mae / max_mae * 20 + 0.5)
        print(f"  {lbl:<10} {mae:>10.6f}  {bar}")
    print(f"{'='*56}\n")


def plot_scatter(results: dict, out_path: Path) -> None:
    """
    2×3 grid of pred-vs-true scatter plots, one per Voigt component.
    A perfect model has all points on the diagonal line.
    """
    preds = results["preds"]      # (N, 6)
    targets = results["targets"]  # (N, 6)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, (ax, lbl) in enumerate(zip(axes, VOIGT_LABELS)):
        p = preds[:, i]
        t = targets[:, i]
        mae = float(np.abs(p - t).mean())

        # Subsample for speed if N is large
        if len(p) > 2000:
            idx = np.random.default_rng(0).choice(len(p), 2000, replace=False)
            p, t = p[idx], t[idx]

        lo = min(p.min(), t.min())
        hi = max(p.max(), t.max())
        pad = (hi - lo) * 0.05 or 1e-4

        ax.scatter(t, p, s=6, alpha=0.4, linewidths=0, color="steelblue")
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "r--", lw=1.2, label="ideal")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("True", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(f"{lbl}  (MAE={mae:.4f})", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Predicted vs True — Voigt Strain Components\n"
        f"N={results['n_samples']}  Overall MAE={results['overall_mae']:.4f}  RMSE={results['rmse']:.4f}",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved to: {out_path}")


def print_spot_checks(
    X: np.ndarray,
    y_strain: np.ndarray,
    model: torch.nn.Module,
    norm_stats: dict,
    norm_method: str,
    device: torch.device,
    n: int = 5,
) -> None:
    print(f"\n  Spot-check (first {n} samples):")
    print(f"  {'idx':>4}  {'mean |err|':>11}  {'pred (ε11..ε12)':>40}  true (ε11..ε12)")
    for i in range(min(n, len(X))):
        xb = preprocess_batch(X[i : i + 1], norm_stats, norm_method).to(device)
        with torch.no_grad():
            pred = model(xb)["strain"].cpu().numpy()[0]
        gt = y_strain[i]
        mae_i = float(np.abs(pred - gt).mean())
        pred_str = " ".join(f"{v:+.4f}" for v in pred)
        true_str = " ".join(f"{v:+.4f}" for v in gt)
        print(f"  {i:>4}  {mae_i:>11.6f}  {pred_str}  {true_str}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Hydra run directory. Default: latest under outputs/.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Directory containing pattern and strain .npy files.")
    parser.add_argument("--patterns-file", default="X_patterns.npy")
    parser.add_argument("--strain-file", default="y_strain.npy")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate on first N samples only (default: all).")
    parser.add_argument("--spot-checks", type=int, default=5,
                        help="Number of individual samples to print (default: 5).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip saving the scatter plot figure.")
    args = parser.parse_args()

    outputs_root = Path("outputs")
    run_dir = args.run_dir if args.run_dir else find_latest_run(outputs_root)
    print(f"Run directory : {run_dir}")

    cfg, norm_method, norm_stats, ckpt_path = load_run(run_dir, args.checkpoint)
    print(f"Checkpoint    : {ckpt_path}")
    print(f"Norm          : {norm_method}  stats={norm_stats}")

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
    y_strain = np.load(args.data_dir / args.strain_file)
    print(f"Patterns : {X.shape}  dtype={X.dtype}")
    print(f"Strain   : {y_strain.shape}  dtype={y_strain.dtype}")

    assert len(X) == len(y_strain), "Pattern and strain arrays must have the same length."

    # ── Spot checks ───────────────────────────────────────────────────────
    print_spot_checks(X, y_strain, model, norm_stats, norm_method, device, n=args.spot_checks)

    # ── Full eval ─────────────────────────────────────────────────────────
    results = run_eval(
        model, X, y_strain, norm_stats, norm_method, device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    print_results(results)

    # ── Scatter plot ──────────────────────────────────────────────────────
    if not args.no_plot:
        plot_scatter(results, run_dir / "eval_scatter.png")

    # ── Save results ──────────────────────────────────────────────────────
    out_path = run_dir / "eval_results.json"
    saveable = {k: v for k, v in results.items() if k not in ("preds", "targets")}
    out_path.write_text(json.dumps(saveable, indent=2))
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
