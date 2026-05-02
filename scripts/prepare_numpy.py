"""
Convert a raw EBSD HDF5 file to .npy files ready for training.

Usage:
    python scripts/prepare_numpy.py --h5 /path/to/data.h5 --out data/my_dataset/
    python scripts/prepare_numpy.py --h5 /path/to/data.h5 --out data/my_dataset/ --force

Produces:
    X_patterns.npy    (N, H, W)   float32   — raw pattern images
    y_euler.npy       (N, 3)      float64   — Euler angles in degrees
    y_quaternion.npy  (N, 4)      float64   — unit quaternions
    y_strain.npy      (N, 6)      float64   — Voigt strain (if strain data present)

Convention: always pass a named subdirectory as --out so datasets are isolated and
never accidentally overwritten, e.g.:
    --out data/stage1_10k_multiaxial/
    --out data/spatial_100x100_combined/
Without --force the script refuses to write into a directory that already contains
any of these files, protecting existing datasets.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from lbp_kikuchi.data.loader import EBSDDataLoader

_PROTECTED_FILES = ["X_patterns.npy", "y_strain.npy", "y_euler.npy", "y_quaternion.npy"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EBSD HDF5 to .npy files for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--h5", required=True, help="Path to EBSD HDF5 file.")
    parser.add_argument(
        "--out", default="data/", help="Output directory (default: data/)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .npy files. Without this flag the script exits "
             "if any target file already exists.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Overwrite guard ────────────────────────────────────────────────────────
    if not args.force:
        existing = [f for f in _PROTECTED_FILES if (out_dir / f).exists()]
        if existing:
            print(
                f"\n[ERROR] The following files already exist in {out_dir.resolve()}:\n"
                + "".join(f"  {f}\n" for f in existing)
                + "\nPassing --force will overwrite them.\n"
                "To keep both datasets, choose a different --out directory, e.g.:\n"
                f"  --out data/my_new_dataset/"
            )
            sys.exit(1)

    loader = EBSDDataLoader(args.h5)
    print(loader)
    loader.to_numpy(str(out_dir))

    print(f"\nDone. Files written to: {out_dir.resolve()}")
    print("Next steps:")
    print("  Stage 1 : python scripts/train_encoder.py data.path=" + str(out_dir))
    print("  Stage 2 : python scripts/train_grid.py    data.path=" + str(out_dir)
          + " data.grid_rows=? data.grid_cols=?")
    print("  Stage 3 : python scripts/train_pair.py    data.path=" + str(out_dir)
          + " data.grid_rows=? data.grid_cols=?")


if __name__ == "__main__":
    main()
