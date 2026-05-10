"""
Validation and sanity checks for generated .npy datasets.

Run after the full pipeline to confirm that all output files are
well-formed before handing off to the training pipeline.
"""

import os
import numpy as np


# Required files that must always be present.
_EXPECTED = {
    "X_patterns.npy":   {"ndim": 4, "shape_suffix": (None, None), "dtype": np.float32},
    "y_strain.npy":     {"ndim": 2, "shape_suffix": (6,),          "dtype": np.float64},
    "y_quaternion.npy": {"ndim": 2, "shape_suffix": (4,),          "dtype": np.float64},
    "y_euler.npy":      {"ndim": 2, "shape_suffix": (3,),          "dtype": np.float64},
}

# Optional files present only for spatial (Stage 2/3) datasets.
_OPTIONAL = {
    "y_positions.npy":  {"ndim": 2, "shape_suffix": (2,),          "dtype": np.int32},
}


class ValidationError(Exception):
    pass


def validate_processed_dir(processed_dir: str, strict: bool = True) -> dict:
    """
    Validate all .npy files in `processed_dir`.

    Checks:
        1. All required files are present.
        2. Arrays have the expected ndim and last-dimension sizes.
        3. The leading dimension N is consistent across all files.
        4. No NaN or Inf values.
        5. Quaternions are unit-normalised (|q| ≈ 1).
        6. Pattern values are finite and non-negative.
        7. Strain values are in a physically plausible range (|ε| < 0.5).

    Args:
        processed_dir: Directory containing X_patterns.npy, y_*.npy.
        strict:        If True, raise ValidationError on any failure.
                       If False, collect warnings and return them.

    Returns:
        Summary dict with keys: n_patterns, pattern_shape, warnings, passed.
    """
    warnings = []
    summary = {}

    def _warn(msg: str):
        warnings.append(f"  WARNING: {msg}")
        if strict:
            raise ValidationError(msg)

    # ── 1. File presence ─────────────────────────────────────────────────────
    for fname in _EXPECTED:
        fpath = os.path.join(processed_dir, fname)
        if not os.path.exists(fpath):
            _warn(f"Missing file: {fpath}")

    # ── Load ─────────────────────────────────────────────────────────────────
    arrays = {}
    for fname in _EXPECTED:
        fpath = os.path.join(processed_dir, fname)
        if os.path.exists(fpath):
            arrays[fname] = np.load(fpath)

    # Optional files — load silently if present; validate if found.
    for fname in _OPTIONAL:
        fpath = os.path.join(processed_dir, fname)
        if os.path.exists(fpath):
            arrays[fname] = np.load(fpath)

    if not arrays:
        raise ValidationError(f"No .npy files found in {processed_dir}")

    # ── 2. Shape / dtype ─────────────────────────────────────────────────────
    all_specs = {**_EXPECTED, **_OPTIONAL}
    for fname, arr in arrays.items():
        exp = all_specs.get(fname)
        if exp is None:
            continue
        if arr.ndim != exp["ndim"]:
            _warn(f"{fname}: expected {exp['ndim']}D, got {arr.ndim}D")
        for dim_idx, expected_size in enumerate(exp["shape_suffix"], start=1):
            if expected_size is not None and arr.shape[-dim_idx] != expected_size:
                _warn(f"{fname}: last dim {dim_idx} expected {expected_size}, got {arr.shape[-dim_idx]}")
        if arr.dtype != exp["dtype"]:
            _warn(f"{fname}: expected dtype {exp['dtype'].__name__}, got {arr.dtype}")

    # ── 3. Consistent N ──────────────────────────────────────────────────────
    ns = {fname: arr.shape[0] for fname, arr in arrays.items()}
    unique_ns = set(ns.values())
    if len(unique_ns) > 1:
        _warn(f"Inconsistent N across files: {ns}")
    n_patterns = ns.get("X_patterns.npy", list(ns.values())[0])
    summary["n_patterns"] = n_patterns

    # ── 4. NaN / Inf ─────────────────────────────────────────────────────────
    for fname, arr in arrays.items():
        if not np.isfinite(arr).all():
            n_bad = (~np.isfinite(arr)).sum()
            _warn(f"{fname}: {n_bad} non-finite (NaN/Inf) values")

    # ── 5. Quaternion unit norm ───────────────────────────────────────────────
    if "y_quaternion.npy" in arrays:
        q = arrays["y_quaternion.npy"]
        norms = np.linalg.norm(q, axis=1)
        max_dev = np.abs(norms - 1.0).max()
        if max_dev > 1e-5:
            _warn(f"y_quaternion.npy: max |‖q‖ − 1| = {max_dev:.2e} (not unit quaternions)")

    # ── 6. Patterns non-negative ──────────────────────────────────────────────
    if "X_patterns.npy" in arrays:
        x = arrays["X_patterns.npy"]
        if x.min() < 0:
            _warn(f"X_patterns.npy: min value {x.min():.4f} < 0 (unexpected for EBSD)")
        summary["pattern_shape"] = x.shape[1:]
        summary["intensity_range"] = (float(x.min()), float(x.max()))

    # ── 7. Strain range ───────────────────────────────────────────────────────
    if "y_strain.npy" in arrays:
        eps = arrays["y_strain.npy"]
        abs_max = np.abs(eps).max()
        if abs_max > 0.5:
            _warn(f"y_strain.npy: max |ε| = {abs_max:.4f} — unusually large (>50% strain)")
        summary["strain_range"] = (float(eps.min()), float(eps.max()))

    # ── 8. Scan-position sanity (spatial mode only) ───────────────────────────
    if "y_positions.npy" in arrays:
        pos = arrays["y_positions.npy"]
        N_pos = pos.shape[0]
        N_ref = n_patterns
        if N_pos != N_ref:
            _warn(f"y_positions.npy: has {N_pos} rows but N={N_ref}")
        if pos.min() < 0:
            _warn(f"y_positions.npy: negative position values found (min={pos.min()})")
        rows_seen = np.unique(pos[:, 0])
        cols_seen = np.unique(pos[:, 1])
        summary["scan_rows"] = int(rows_seen.max() + 1)
        summary["scan_cols"] = int(cols_seen.max() + 1)
        expected_N = summary["scan_rows"] * summary["scan_cols"]
        if expected_N != N_ref:
            _warn(
                f"y_positions.npy: grid {summary['scan_rows']}×{summary['scan_cols']}"
                f" = {expected_N} ≠ N={N_ref}"
            )

    summary["warnings"] = warnings
    summary["passed"] = len(warnings) == 0
    return summary


def print_summary(processed_dir: str) -> None:
    """Print a human-readable validation report to stdout."""
    print(f"\n{'='*60}")
    print(f"  Dataset validation: {processed_dir}")
    print(f"{'='*60}")

    try:
        result = validate_processed_dir(processed_dir, strict=False)
    except ValidationError as e:
        print(f"  FATAL: {e}")
        return

    print(f"  N patterns    : {result.get('n_patterns', '?'):,}")
    if "pattern_shape" in result:
        *_, h, w = result["pattern_shape"]   # works for (H,W) or (C,H,W)
        print(f"  Pattern size  : {h} × {w}  px")
    if "intensity_range" in result:
        lo, hi = result["intensity_range"]
        print(f"  Intensity     : [{lo:.4f}, {hi:.4f}]")
    if "strain_range" in result:
        lo, hi = result["strain_range"]
        print(f"  Strain Voigt  : [{lo:.6f}, {hi:.6f}]")
    if "scan_rows" in result:
        print(f"  Scan grid     : {result['scan_rows']} × {result['scan_cols']}  (spatial mode)")

    if result["passed"]:
        print(f"\n  ✓ All checks passed.")
    else:
        print(f"\n  Warnings ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(w)

    print(f"{'='*60}\n")
