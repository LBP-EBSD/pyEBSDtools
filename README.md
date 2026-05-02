# EMsoft EBSD Python Tools

Python utilities for working with EMsoft synthetic EBSD pattern data. Use these
to visualize patterns, load data for ML training, and generate bulk orientation/strain files.

## Setup

```bash
pip install -r requirements.txt
```

## What Is EBSD Data?

**EBSD (Electron Backscatter Diffraction)** is a scanning electron microscopy technique
that produces diffraction patterns — like crystal "fingerprints". Each pattern is a
**480×640 pixel grayscale image** showing Kikuchi bands that encode the crystal's:

- **Orientation** — which way the crystal lattice is pointing (3 Euler angles)
- **Elastic strain** — how the lattice is stretched or compressed (deformation tensor)

```
 ┌──────────────────────┐
 │  \    /\    /\      │   ← Kikuchi bands (diffraction)
 │   \  /  \  /  \     │     The positions/shapes of bands
 │    \/    \/    \    │     encode crystal orientation
 │     EBSD Pattern     │     and elastic strain.
 │   480 × 640 pixels  │
 └──────────────────────┘
```

### What You Get from EMsoft

| Dataset | Shape | Description |
|---|---|---|
| `patterns` | `(N, 480, 640)` | N grayscale pattern images |
| `euler_angles` | `(N, 3)` | Bunge Euler angles [φ₁, Φ, φ₂] in degrees |
| `F_tensors` | `(N, 3, 3)` | Deformation tensors F = I + ε (column-major) |
| `quaternions` | `(N, 4)` | Orientation as [q₀, q₁, q₂, q₃] |
| `voigt_strain` | `(N, 6)` | Elastic strain in Voigt format [ε₁₁,ε₂₂,ε₃₃,ε₂₃,ε₁₃,ε₁₂] |

### Common ML Labels

For **orientation prediction** (predicting orientation from a pattern):
- **Input X**: `(N, 480, 640)` — the pattern images
- **Target y**: `(N, 3)` — Euler angles (regression), or `(N, 4)` — quaternions (regression/classification)

For **strain prediction** (predicting elastic strain from a pattern):
- **Input X**: `(N, 480, 640)` — the pattern images
- **Target y**: `(N, 6)` — Voigt strain tensor

For **joint prediction** (predict both orientation AND strain):
- Use a multi-head model with separate output layers

---

## Scripts

### 1. `scripts/prepare_numpy.py` — HDF5 → NumPy arrays

Converts an EMsoft HDF5 file into `.npy` arrays used by training/inference.

```bash
python scripts/prepare_numpy.py --h5 /path/to/data.h5 --out data/
```

Creates:
- `X_patterns.npy`
- `y_euler.npy`
- `y_quaternion.npy`
- `y_strain.npy` (if strain exists in source)

### 2. `scripts/visualize.py` — Pattern visualization

```bash
# Run default set of plots
python scripts/visualize.py /path/to/data.h5

# Specific views
python scripts/visualize.py /path/to/data.h5 --grid
python scripts/visualize.py /path/to/data.h5 --single 1
python scripts/visualize.py /path/to/data.h5 --compare 1 2
python scripts/visualize.py /path/to/data.h5 --heatmap
python scripts/visualize.py /path/to/data.h5 --histogram
python scripts/visualize.py /path/to/data.h5 --export-tiff
python scripts/visualize.py /path/to/data.h5 --export-png
```

### 3. `scripts/train_encoder.py` — Train (Hydra)

```bash
python scripts/train_encoder.py
python scripts/train_encoder.py training.epochs=100 training.lr=5e-4
python scripts/train_encoder.py data.path=data/overfit64 training.test_split=0.0
```

Important behavior:
- Uses `train/val/test` split from config (`training.val_split`, `training.test_split`)
- Computes normalisation stats from **train split only**
- Saves:
  - `checkpoints/norm_stats.json`
  - `checkpoints/split_indices.json`
  - `checkpoints/best.pt`, `checkpoints/last.pt`

### 4. `scripts/infer_eval.py` — Split-aware evaluation

Defaults:
- latest run under `outputs/`
- `--split val`

```bash
python scripts/infer_eval.py
python scripts/infer_eval.py --run-dir outputs/2026-04-29/14-00-58 --split val
python scripts/infer_eval.py --split train
python scripts/infer_eval.py --split test
python scripts/infer_eval.py --split all
```

Flags:
- `--run-dir`
- `--split` (`train|val|test|all`)
- `--data-dir` (default: from run config)
- `--checkpoint` (`best.pt|last.pt`)
- `--batch-size`
- `--max-samples`
- `--spot-checks`
- `--no-plot`

Outputs:
- `eval_scatter_<split>.png`
- `eval_results_<split>.json`

### 5. `scripts/infer.py` — Single-sample inference

```bash
python scripts/infer.py
python scripts/infer.py --index 12
python scripts/infer.py --run-dir outputs/2026-04-29/14-00-58 --save-plot pred.png
```

Flags:
- `--run-dir`
- `--data-dir` (default: from run config)
- `--patterns-file`
- `--index`
- `--checkpoint`
- `--save-plot`

### 6. `scripts/generate_angles.py` — Generate orientations & strains

This generates the input file for EMsoft's `EMEBSD` pattern generator. Run this
**inside the Docker container** after installing Python packages, or run on the host
and copy the output to the container.

```bash
# Generate 10,000 random orientations, no strain (for pure orientation learning)
python generate_angles.py -n 10000 -o ./orientations.txt

# Generate 5,000 patterns with random uniaxial strain (0-2%)
python generate_angles.py -n 5000 -s 0.02 --strain-type multiaxial

# Generate 1,000 patterns with uniform hydrostatic strain (0.5%)
python generate_angles.py -n 1000 --uniform-strain 0.005

# Pure orientations only (no strain, for anglefiletype='orientations')
python generate_angles.py -n 1000 --orient-only

# Random mixed strain types with reproducibility
python generate_angles.py -n 10000 --strain-type random -s 0.015 --seed 42
```

**Available strain types:**
| Type | Description | Use case |
|---|---|---|
| `uniform` | No strain (identity F) | Orientation learning baseline |
| `uniaxial_x/y/z` | Tension/compression along one axis | Tensile testing ML |
| `biaxial_xy/yz/xz` | Equal strain in two axes | Biaxial loading |
| `multiaxial` | Random strain in all 6 components | General strain prediction |
| `shear_xy/xz/yz` | Pure shear deformation | Shear strain ML |
| `random` | Mix of above (50% multiaxial, rest distributed) | General training |

---

## Quick Start

### Step 1: Convert raw HDF5 data

```bash
pip install -r requirements.txt
python scripts/prepare_numpy.py --h5 ~/EMsoftData/Fe_FCC_exp/Fe_EBSD_patterns.h5 --out data/
```

### Step 2: (Optional) Visualize patterns

```bash
python scripts/visualize.py ~/EMsoftData/Fe_FCC_exp/Fe_EBSD_patterns.h5 --grid --heatmap
```

### Step 3: Train

```python
python scripts/train_encoder.py
```

### Step 4: Evaluate held-out split

```bash
python scripts/infer_eval.py --split val
# Optional:
python scripts/infer_eval.py --split train
```

---

## Data Format Reference

### HDF5 File Structure

```
your_file.h5
├── EMData/
│   ├── EBSD/
│   │   ├── EBSDPatterns      (N, 480, 640) float32  ← pattern images
│   │   ├── EulerAngles       (N, 3)      float64  ← [phi1, Phi, phi2] in degrees
│   │   ├── numangles         scalar int
│   │   └── xtalname          string
│   └── ...
├── EMheader/
│   └── EBSD/                 ← metadata
└── NMLfiles/                 ← input parameters saved
```

### Euler Angles (Bunge Convention)

Three rotation angles in degrees:
- **φ₁** (phi1): rotation about Z axis
- **Φ** (Phi): rotation about X' axis
- **φ₂** (phi2): rotation about Z'' axis

All angles in range [0°, 360°]. Convert to quaternions with `loader.get_quaternions()`.

### Deformation Tensor F

F = I + ε, where ε is the elastic strain tensor. Stored in **column-major order**:

```
F = | F11 F12 F13 |
    | F21 F22 F23 |
    | F31 F32 F33 |
```

In the file: `F11 F21 F31 F12 F22 F32 F13 F23 F33`

### Voigt Strain Format

Six-component strain tensor:
```
[ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂]
  xx   yy   zz   yz   xz   xy
```

For small strains: F ≈ I + ε → just subtract identity from F to get ε.

---

## Expected Generation Times

| N Patterns | EMEBSD Time | Use Case |
|---|---|---|
| 100 | ~2 min | Quick testing |
| 1,000 | ~15 min | Small dataset |
| 10,000 | ~2-3 hours | Moderate training set |
| 100,000 | ~15-30 hours | Full training set |

> **Tip:** For large datasets, consider running overnight. The EMEBSD step is
> CPU-only and writes incrementally to HDF5 — you can monitor progress from
> the terminal output.

---

## Troubleshooting

**Q: `h5py not found`**
```bash
pip install h5py numpy matplotlib Pillow
```

**Q: Patterns look all black/white**
- Normalize using `loader.get_normalized('minmax')` or `loader.get_normalized('zscore')`
- The raw float32 values may be in a narrow range (e.g., 0.01–0.20)

**Q: Want to change pattern size?**
- Adjust `numsx`, `numsy`, `binning` in the EMEBSD.nml file (in Docker)
- Re-run `EMEBSD` after changing

**Q: Different crystal structure?**
- Replace `Ni.xtal` with any `.xtal` file from EMsoftData
- Re-run the full pipeline (EMMCOpenCL → EMEBSDmaster → EMEBSD)
