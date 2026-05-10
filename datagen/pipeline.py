#!/usr/bin/env python3
"""
Data generation pipeline — master orchestrator.

Reads config.yaml and runs all four stages in sequence:

    Stage 1 — Sample     datagen/sampler.py    → datagen/angle_generation.py
    Stage 2 — Simulate   datagen/emsoft.py     EMsoft inside Docker → .h5
    Stage 3 — Convert    datagen/convert.py    .h5 + labels → .npy
    Stage 4 — Validate   helpers/validate.py   sanity checks

Usage:
    python datagen/pipeline.py                              # uses datagen/configs/config.yaml
    python datagen/pipeline.py --config datagen/configs/my.yaml
    python datagen/pipeline.py --skip-simulate              # if .h5 already exists
    python datagen/pipeline.py --validate-only              # re-validate existing output

Run from repo root:
    make generate
    make generate CONFIG=datagen/configs/my_experiment.yaml
"""

import argparse
import os
import sys
import time
import yaml

# Make sure imports work from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import sampler as sampler_mod
from datagen import emsoft  as emsoft_mod
from datagen import convert as convert_mod
from helpers import validate as validate_mod


def load_config(config_path: str) -> dict:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    required_sections = ["generation", "emsoft", "docker", "paths"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"config.yaml missing required section: [{section}]")

    required_paths = ["data_dir", "xtal_dir", "processed_dir", "experiment_name"]
    for key in required_paths:
        if key not in cfg["paths"]:
            raise ValueError(f"config.yaml paths.{key} is required")

    gen = cfg["generation"]
    if gen.get("spatial_field"):
        for key in ("grid_rows", "grid_cols"):
            if key not in gen:
                raise ValueError(
                    f"config.yaml generation.{key} is required when generation.spatial_field is true"
                )


def run_pipeline(
    cfg: dict,
    skip_sample: bool   = False,
    skip_simulate: bool = False,
    skip_convert: bool  = False,
    validate_only: bool = False,
) -> None:
    """
    Run the full data generation pipeline.

    Args:
        cfg:            Parsed config dict.
        skip_sample:    Skip Stage 1 (assume angle files already exist).
        skip_simulate:  Skip Stage 2 (assume .h5 already exists).
        skip_convert:   Skip Stage 3 (assume .npy files already exist).
        validate_only:  Only run Stage 4.
    """
    t_start = time.time()
    paths    = cfg["paths"]
    data_dir = os.path.expanduser(paths["data_dir"])
    exp_dir  = os.path.join(data_dir, paths["experiment_name"])
    proc_dir = os.path.expanduser(paths["processed_dir"])

    _header("Data Generation Pipeline")
    _print_config_summary(cfg)

    # ── Save config snapshot alongside the experiment data ────────────────────
    if not validate_only:
        os.makedirs(exp_dir, exist_ok=True)
        snapshot_path = os.path.join(exp_dir, "config_snapshot.yaml")
        with open(snapshot_path, "w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
        print(f"\n  Config snapshot → {snapshot_path}")

    # ── Stage 4 only ─────────────────────────────────────────────────────────
    if validate_only:
        _stage("4", "Validate")
        validate_mod.print_summary(proc_dir)
        return

    # ── Stage 1: Sample ───────────────────────────────────────────────────────
    sampler_paths = None
    if not skip_sample:
        _stage("1", "Sample — generating orientations + F tensors")
        sampler_paths = sampler_mod.run_from_config(cfg)
    else:
        _stage("1", "Sample — SKIPPED")
        exp_name = paths["experiment_name"]
        sampler_paths = {
            "angles_txt":    os.path.join(exp_dir, f"{exp_name}_angles.txt"),
            "ftensors_npy":  os.path.join(exp_dir, f"{exp_name}_Ftensors.npy"),
            "euler_npy":     os.path.join(exp_dir, f"{exp_name}_euler.npy"),
        }
        _check_files_exist(sampler_paths)
        # Include positions file if it exists (spatial mode only).
        pos_path = os.path.join(exp_dir, f"{exp_name}_positions.npy")
        if os.path.exists(pos_path):
            sampler_paths["positions_npy"] = pos_path

    # ── Stage 2: Simulate (EMsoft/Docker) ────────────────────────────────────
    if not skip_simulate:
        _stage("2", "Simulate — writing NML files + running EMsoft in Docker")
        # Write NML files first so they live alongside the data
        writer = emsoft_mod.NMLWriter(cfg)
        writer.write_all(exp_dir)

        # Docker runs as EMuser (uid=501, gid=3000); the exp_dir was created by the
        # host user and isn't writable by others by default — chmod it first.
        os.chmod(exp_dir, 0o777)

        runner = emsoft_mod.DockerRunner(cfg)
        runner.ensure_image()
        runner.write_container_config()
        runner.run_pipeline()
    else:
        _stage("2", "Simulate — SKIPPED")
        h5_path = os.path.join(exp_dir, "Fe_EBSD_patterns.h5")
        if not os.path.exists(h5_path):
            print(f"  [ERROR] Expected .h5 not found: {h5_path}")
            sys.exit(1)

    # ── Stage 3: Convert ─────────────────────────────────────────────────────
    if not skip_convert:
        _stage("3", "Convert — HDF5 + labels → .npy")
        convert_mod.run_from_config(cfg, sampler_paths)
    else:
        _stage("3", "Convert — SKIPPED")

    # ── Stage 4: Validate ────────────────────────────────────────────────────
    _stage("4", "Validate")
    validate_mod.print_summary(proc_dir)

    elapsed = time.time() - t_start
    print(f"\nPipeline complete in {elapsed / 60:.1f} min.\n")


# ─── Printing helpers ─────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _stage(num: str, description: str) -> None:
    print(f"\n── Stage {num}: {description}")
    print("   " + "─" * 50)


def _print_config_summary(cfg: dict) -> None:
    gen   = cfg["generation"]
    paths = cfg["paths"]
    spatial = gen.get("spatial_field", False)
    if spatial:
        n_patterns = gen.get("grid_rows", "?") * gen.get("grid_cols", "?")
        mode_line = (f"  Mode        : spatial  {gen.get('grid_rows')}×{gen.get('grid_cols')}"
                     f"  field={gen.get('field_type', 'combined')}")
    else:
        n_patterns = gen.get("n_patterns", "?")
        mode_line = (f"  Strain type : {gen.get('strain_type', 'uniform')}"
                     f"  (mag={gen.get('strain_magnitude', 0.0)})")
    print(f"\n  Experiment  : {paths['experiment_name']}")
    print(f"  N patterns  : {n_patterns:,}" if isinstance(n_patterns, int) else f"  N patterns  : {n_patterns}")
    print(mode_line)
    print(f"  Seed        : {gen.get('seed', 'None')}")
    print(f"  Docker image: {cfg['docker']['image']}")
    print(f"  Raw out     : {paths['data_dir']}/{paths['experiment_name']}")
    print(f"  Processed   : {paths['processed_dir']}")


def _check_files_exist(paths_dict: dict) -> None:
    for role, path in paths_dict.items():
        if not os.path.exists(path):
            print(f"  [ERROR] Expected file not found ({role}): {path}")
            sys.exit(1)
        print(f"  Found {role}: {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EMsoft EBSD data generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="datagen/configs/config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--skip-sample", action="store_true",
        help="Skip Stage 1 — reuse existing angle + label files.",
    )
    parser.add_argument(
        "--skip-simulate", action="store_true",
        help="Skip Stage 2 — reuse existing .h5 file (don't run Docker/EMsoft).",
    )
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="Skip Stage 3 — reuse existing .npy files.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only run Stage 4 validation on existing processed data.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        print(f"  Copy config.yaml.example to config.yaml and edit it.")
        sys.exit(1)

    cfg = load_config(args.config)

    run_pipeline(
        cfg,
        skip_sample   = args.skip_sample,
        skip_simulate = args.skip_simulate,
        skip_convert  = args.skip_convert,
        validate_only = args.validate_only,
    )


if __name__ == "__main__":
    main()
