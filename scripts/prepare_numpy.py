"""
Convert a raw EBSD HDF5 file to .npy files ready for training.

Usage:
    python scripts/prepare_numpy.py --h5 /path/to/data.h5 --out data/processed/

Produces:
    X_patterns.npy    (N, H, W)   float32   — raw pattern images
    y_euler.npy       (N, 3)      float64   — Euler angles in degrees
    y_quaternion.npy  (N, 4)      float64   — unit quaternions
    y_strain.npy      (N, 6)      float64   — Voigt strain (if strain data present)
"""

import argparse
from pathlib import Path

from lbp_kikuchi.data.loader import EBSDDataLoader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EBSD HDF5 to .npy files for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--h5", required=True, help="Path to EBSD HDF5 file.")
    parser.add_argument(
        "--out", default="data/processed/", help="Output directory (default: data/processed/)."
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = EBSDDataLoader(args.h5)
    print(loader)
    loader.to_numpy(str(out_dir))

    print(f"\nDone. Files written to: {out_dir.resolve()}")
    print("Next: run training with  python scripts/train_encoder.py")


if __name__ == "__main__":
    main()
