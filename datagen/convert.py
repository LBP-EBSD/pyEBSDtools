"""
HDF5 → .npy converter — Stage 3 of the data generation pipeline.

Reads:
  - <exp_dir>/Fe_EBSD_patterns.h5       — EMsoft output (patterns + Euler angles)
  - <exp_dir>/<experiment_name>_Ftensors.npy  — F tensors saved by sampler
  - <exp_dir>/<experiment_name>_euler.npy     — Euler angles saved by sampler

Writes to processed_dir:
  - X_patterns.npy     (N, 1, H, W)  float32  — raw patterns (channel-first)
  - y_strain.npy       (N, 6)     float64  — Voigt strain
  - y_quaternion.npy   (N, 4)     float64  — unit quaternions
  - y_euler.npy        (N, 3)     float64  — Euler angles in degrees

This module decouples the training pipeline from the raw HDF5 format and from
whatever EMsoft may or may not write back into its output file. Labels always
come from the sampler's saved .npy files, never from the HDF5.
"""

import os
import numpy as np

from helpers import hdf5_io, crystal


def convert(
    h5_path: str,
    ftensors_npy: str,
    euler_npy: str,
    out_dir: str,
) -> dict[str, str]:
    """
    Convert one EMsoft HDF5 file + sampler labels into training-ready .npy files.

    Args:
        h5_path:      Path to Fe_EBSD_patterns.h5.
        ftensors_npy: Path to <exp>_Ftensors.npy from sampler.
        euler_npy:    Path to <exp>_euler.npy from sampler.
        out_dir:      Output directory for .npy files.

    Returns:
        dict mapping dataset role → absolute output path.
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"[convert] Reading patterns from: {h5_path}")
    patterns = hdf5_io.read_patterns(h5_path)            # (N, H, W) float32
    N_h5 = patterns.shape[0]
    print(f"[convert]   patterns shape: {patterns.shape}  dtype: {patterns.dtype}")

    # ── Load saved labels ────────────────────────────────────────────────────
    print(f"[convert] Loading F tensors from: {ftensors_npy}")
    F_tensors = np.load(ftensors_npy)                    # (N, 3, 3) float64
    print(f"[convert] Loading Euler angles from: {euler_npy}")
    euler = np.load(euler_npy)                           # (N, 3) float64

    # ── Consistency check ────────────────────────────────────────────────────
    N_labels = len(F_tensors)
    if N_h5 != N_labels:
        raise ValueError(
            f"Pattern count mismatch: HDF5 has {N_h5} patterns "
            f"but labels have {N_labels} entries. "
            f"Did you use the right labels file for this HDF5?"
        )

    # ── Derive labels ─────────────────────────────────────────────────────────
    print(f"[convert] Computing Voigt strain from F tensors...")
    voigt_strain = crystal.ftensor_to_voigt(F_tensors)   # (N, 6)

    print(f"[convert] Computing quaternions from Euler angles...")
    quaternions  = crystal.euler_to_quaternion(euler)    # (N, 4)

    # Validate quaternion norms
    norms = np.linalg.norm(quaternions, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"Quaternion normalisation failed: max dev = {np.abs(norms - 1.0).max():.2e}"

    # ── Write outputs ─────────────────────────────────────────────────────────
    # Add channel dim: (N, H, W) → (N, 1, H, W) as expected by the ML pipeline
    outputs = {
        "X_patterns.npy":   (patterns[:, np.newaxis].astype(np.float32), "float32"),
        "y_strain.npy":     (voigt_strain.astype(np.float64), "float64"),
        "y_quaternion.npy": (quaternions.astype(np.float64),  "float64"),
        "y_euler.npy":      (euler.astype(np.float64),        "float64"),
    }

    paths = {}
    print(f"\n[convert] Writing to: {out_dir}")
    for fname, (arr, _dtype) in outputs.items():
        path = os.path.join(out_dir, fname)
        np.save(path, arr)
        paths[fname] = path
        print(f"  {fname:<22} shape={str(arr.shape):<15} dtype={arr.dtype}")

    print(f"\n[convert] Done. {N_h5:,} patterns written.")
    return paths


def run_from_config(cfg: dict, sampler_paths: dict) -> dict[str, str]:
    """
    Run conversion from a parsed config dict + the paths returned by sampler.save().

    Args:
        cfg:           Parsed config.yaml dict.
        sampler_paths: Dict from datagen/sampler.run_from_config() with keys
                       'angles_txt', 'ftensors_npy', 'euler_npy'.

    Returns:
        Dict of output .npy paths.
    """
    paths    = cfg["paths"]
    data_dir = os.path.expanduser(paths["data_dir"])
    exp_name = paths["experiment_name"]
    exp_dir  = os.path.join(data_dir, exp_name)

    h5_path  = os.path.join(exp_dir, "Fe_EBSD_patterns.h5")
    out_dir  = os.path.expanduser(paths["processed_dir"])

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"EMsoft output not found: {h5_path}\n"
            f"Run `make generate` or run EMsoft manually first."
        )

    return convert(
        h5_path      = h5_path,
        ftensors_npy = sampler_paths["ftensors_npy"],
        euler_npy    = sampler_paths["euler_npy"],
        out_dir      = out_dir,
    )
