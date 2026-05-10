"""
Orientation and strain sampler — Stage 1 of the data generation pipeline.

Delegates to ``datagen.angle_generation`` (``AngleStrainGenerator``,
``SpatialFieldGenerator``) so the root ``generate_angles.py`` CLI and
``make generate`` share one implementation.

Writes:
  - <experiment_name>_angles.txt      — EMEBSD input file (orpcdef format)
  - <experiment_name>_Ftensors.npy   — (N, 3, 3) F tensors saved as labels
  - <experiment_name>_euler.npy      — (N, 3) Euler angles saved as labels
  - <experiment_name>_positions.npy  — (N, 2) [row, col] scan positions
                                        (spatial mode only; row-major C order)
  - <experiment_name>_metadata.json  — grid dims + generation settings

EMsoft reads ``orpcdef`` with **no** comment line after N — each data line must
be exactly 15 floats (euler ×3, xpc, ypc, L, F column-major ×9).
"""

from __future__ import annotations

import json
import os

import numpy as np

from datagen.angle_generation import AngleStrainGenerator, SpatialFieldGenerator


def run_from_config(cfg: dict) -> dict[str, str]:
    """
    Run Stage 1 from a parsed config dict (as loaded from config.yaml).

    Returns paths dict for angles_txt, ftensors_npy, euler_npy, and
    (spatial mode only) positions_npy.
    """
    gen_cfg = cfg["generation"]
    paths_cfg = cfg["paths"]
    ems = cfg["emsoft"]

    data_dir = os.path.expanduser(paths_cfg["data_dir"])
    exp_dir = os.path.join(data_dir, paths_cfg["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)

    exp_name = paths_cfg["experiment_name"]
    angles_path   = os.path.join(exp_dir, f"{exp_name}_angles.txt")
    ftensor_path  = os.path.join(exp_dir, f"{exp_name}_Ftensors.npy")
    euler_path    = os.path.join(exp_dir, f"{exp_name}_euler.npy")
    pos_path      = os.path.join(exp_dir, f"{exp_name}_positions.npy")
    metadata_path = os.path.join(exp_dir, f"{exp_name}_metadata.json")

    L   = float(ems.get("camera_distance_um", 15000.0))
    xpc = float(gen_cfg.get("xpc", 0.0))
    ypc = float(gen_cfg.get("ypc", 0.0))
    seed = gen_cfg.get("seed")

    spatial = bool(gen_cfg.get("spatial_field", False))

    if spatial:
        rows = int(gen_cfg["grid_rows"])
        cols = int(gen_cfg["grid_cols"])
        n_patterns = rows * cols

        gen_sp = SpatialFieldGenerator(grid_rows=rows, grid_cols=cols, seed=seed)
        orientations, F_tensors, _eps_field = gen_sp.generate(
            field_type=gen_cfg.get("field_type", "combined"),
            scale=float(gen_cfg.get("field_scale", 1.0)),
            constant_orientation=bool(gen_cfg.get("constant_orientation", False)),
            orientation_spread_deg=float(gen_cfg.get("orientation_spread", 1.0)),
            noise_frac=float(gen_cfg.get("noise_frac", 0.05)),
        )

        writer = AngleStrainGenerator(n_patterns=n_patterns, seed=seed)
        writer.set_pattern_center(xpc=xpc, ypc=ypc, L=L)
        # EMsoft: no comment line inside orpcdef (Fortran reads N then N lines of 15 reals)
        writer.write_orpcdef(orientations, F_tensors, angles_path, comment="")

        # ── Scan-grid positions (row-major C order) ────────────────────────
        # Pattern i lives at row = i // cols, col = i % cols.
        rows_arr = np.repeat(np.arange(rows, dtype=np.int32), cols)
        cols_arr = np.tile(np.arange(cols, dtype=np.int32), rows)
        positions = np.stack([rows_arr, cols_arr], axis=1)   # (N, 2)
        np.save(pos_path, positions)

        print(f"[sampler] spatial_field: grid {rows}×{cols} = {n_patterns:,} patterns")

        metadata = {
            "spatial": True,
            "grid_rows": rows,
            "grid_cols": cols,
            "n_patterns": n_patterns,
            "field_type": gen_cfg.get("field_type", "combined"),
            "seed": seed,
        }
    else:
        n_patterns = int(gen_cfg["n_patterns"])
        strain_type = gen_cfg["strain_type"]
        strain_mag = float(gen_cfg["strain_magnitude"])
        uniform_val = gen_cfg.get("uniform_strain")
        if uniform_val is not None:
            strain_type = "uniform"
            uniform_val = float(uniform_val)
        else:
            uniform_val = 0.0

        gen = AngleStrainGenerator(n_patterns=n_patterns, seed=seed)
        gen.set_pattern_center(xpc=xpc, ypc=ypc, L=L)
        gen.set_strain(strain_type, strain_mag, uniform_val)

        orientations, F_tensors = gen.generate(
            strain_type=strain_type,
            strain_magnitude=strain_mag,
            uniform_strain_value=uniform_val,
        )
        gen.summary()
        gen.write_orpcdef(orientations, F_tensors, angles_path, comment="")
        pos_path = None   # no spatial positions for random mode

        metadata = {
            "spatial": False,
            "n_patterns": n_patterns,
            "strain_type": strain_type,
            "strain_magnitude": float(gen_cfg["strain_magnitude"]),
            "seed": seed,
        }

    np.save(ftensor_path, np.asarray(F_tensors, dtype=np.float64))
    np.save(euler_path, np.asarray(orientations, dtype=np.float64))

    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"[sampler] Wrote {orientations.shape[0]:,} samples:")
    print(f"  angles    → {angles_path}")
    print(f"  F tensors → {ftensor_path}  shape={F_tensors.shape}")
    print(f"  euler     → {euler_path}    shape={orientations.shape}")
    if pos_path:
        print(f"  positions → {pos_path}  shape={positions.shape}")
    print(f"  metadata  → {metadata_path}")

    paths = {
        "angles_txt":    angles_path,
        "ftensors_npy":  ftensor_path,
        "euler_npy":     euler_path,
        "metadata_json": metadata_path,
    }
    if pos_path:
        paths["positions_npy"] = pos_path
    return paths
