"""
HDF5 I/O helpers for EMsoft EBSD output files.

All reads are lazy-safe (context-manager based) and return numpy arrays.
Path constants match the actual EMsoft EBSD HDF5 layout.
"""

import os
import numpy as np
import h5py


# ─── HDF5 dataset paths as used by EMsoft ────────────────────────────────────
_PATH_PATTERNS   = "EMData/EBSD/EBSDPatterns"
_PATH_EULER      = "EMData/EBSD/EulerAngles"
_PATH_NUMANGLES  = "EMData/EBSD/numangles"
_PATH_XTALNAME   = "EMData/EBSD/xtalname"
# EMsoft does NOT reliably write deformation tensors back into the HDF5.
# We keep this path here for forward compatibility; use saved labels.npy instead.
_PATH_FTENSOR    = "EMData/EBSD/DeformationTensor"


def read_patterns(h5_path: str) -> np.ndarray:
    """
    Read EBSD patterns from an EMsoft HDF5 file.

    Returns:
        (N, H, W) float32 array — raw pattern images.
    """
    with h5py.File(h5_path, "r") as f:
        if _PATH_PATTERNS not in f:
            raise KeyError(
                f"'EBSDPatterns' not found in {h5_path}. "
                f"Top-level groups: {list(f.keys())}"
            )
        return f[_PATH_PATTERNS][:]


def read_euler(h5_path: str) -> np.ndarray:
    """
    Read Euler angles from an EMsoft HDF5 file.

    Returns:
        (N, 3) float64 array — Bunge Euler angles in degrees [phi1, Phi, phi2].
        Falls back to zeros if the dataset is absent.
    """
    with h5py.File(h5_path, "r") as f:
        if _PATH_EULER in f:
            return np.asarray(f[_PATH_EULER][:], dtype=np.float64)
        n = _read_numangles(f, h5_path)
        return np.zeros((n, 3), dtype=np.float64)


def read_deformation_tensors(h5_path: str) -> np.ndarray | None:
    """
    Try to read deformation tensors from the HDF5.

    EMsoft does not consistently write these back, so this may return None.
    Always prefer loading from the saved labels.npy produced by the sampler.

    Returns:
        (N, 3, 3) float64 array, or None.
    """
    with h5py.File(h5_path, "r") as f:
        if _PATH_FTENSOR in f:
            return np.asarray(f[_PATH_FTENSOR][:], dtype=np.float64)
    return None


def read_xtalname(h5_path: str) -> str:
    """Read the crystal name string stored in the HDF5."""
    with h5py.File(h5_path, "r") as f:
        if _PATH_XTALNAME in f:
            val = f[_PATH_XTALNAME][()]
            return val.decode("utf-8").strip() if isinstance(val, bytes) else str(val).strip()
    return "unknown"


def read_numangles(h5_path: str) -> int:
    """Read the number of patterns stored in the HDF5."""
    with h5py.File(h5_path, "r") as f:
        return _read_numangles(f, h5_path)


def print_tree(h5_path: str) -> None:
    """Print the full dataset tree of an EMsoft HDF5 file (for debugging)."""
    size_mb = os.path.getsize(h5_path) / 1e6
    print(f"\nFile: {h5_path}  ({size_mb:.1f} MB)")
    print("-" * 70)
    print(f"{'Dataset':<50} {'Shape':<20} {'dtype'}")
    print("-" * 70)

    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name:<50} {str(obj.shape):<20} {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"[group] {name}")

    with h5py.File(h5_path, "r") as f:
        f.visititems(_visitor)


# ─── internal ────────────────────────────────────────────────────────────────

def _read_numangles(f: h5py.File, path: str) -> int:
    if _PATH_NUMANGLES in f:
        val = f[_PATH_NUMANGLES][()]
        return int(val[0]) if hasattr(val, "__iter__") else int(val)
    if _PATH_PATTERNS in f:
        return f[_PATH_PATTERNS].shape[0]
    raise RuntimeError(f"Cannot determine numangles from {path}")
