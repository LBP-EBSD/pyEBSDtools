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

### 1. `load_data.py` — Load EBSD Data

```python
from load_data import EBSDDataLoader

# Load from HDF5 file
loader = EBSDDataLoader('/path/to/Fe_EBSD_patterns.h5')
print(loader)  # prints summary

# Get raw data
X = loader.get_patterns()          # (N, 480, 640) images
y = loader.get_euler()             # (N, 3) Euler angles in degrees
q = loader.get_quaternions()       # (N, 4) quaternions
eps = loader.get_voigt_strain()   # (N, 6) Voigt strain or None

# Normalize patterns
X_norm = loader.get_normalized('minmax')   # 0-1 range
X_norm = loader.get_normalized('zscore')   # zero mean, unit variance

# Flatten for classic ML
X_flat = loader.get_flattened()           # (N, 307200)

# Dump to .npy files for easy loading
loader.to_numpy('/output/dir/')
# Creates: X_patterns.npy, y_euler.npy, y_quaternion.npy, y_strain.npy

# TensorFlow dataset
ds = loader.to_tensorflow_dataset(batch_size=32)
```

**As a script** (inspect file structure):
```bash
python load_data.py /path/to/data.h5 --info       # show HDF5 tree
python load_data.py /path/to/data.h5 --dump ./out/  # dump as .npy files
```

### 2. `visualize.py` — Visualize Patterns

```bash
# Run all visualizations (default)
python visualize.py /path/to/data.h5

# Specific views
python visualize.py /path/to/data.h5 --grid              # grid of all patterns
python visualize.py /path/to/data.h5 --single 1           # pattern #1 with profiles
python visualize.py /path/to/data.h5 --compare 1 2       # compare patterns #1 and #2
python visualize.py /path/to/data.h5 --heatmap           # mean/std sensitivity map
python visualize.py /path/to/data.h5 --histogram         # intensity histogram
python visualize.py /path/to/data.h5 --export-tiff        # export as TIFF images
python visualize.py /path/to/data.h5 --export-png         # export as PNG images
python visualize.py /path/to/data.h5 --all                # all of the above

# Save to specific output directory
python visualize.py /path/to/data.h5 -o ./my_output/
```

**As a library** (in Jupyter or code):
```python
from visualize import plot_grid, plot_single_with_profiles, plot_pattern_difference
from load_data import EBSDDataLoader

loader = EBSDDataLoader('/path/to/data.h5')
plot_grid(loader, output_path='grid.png')
plot_single_with_profiles(loader, idx=0, output_path='profile.png')
plot_pattern_difference(loader, idx1=0, idx2=1, output_path='diff.png')
```

### 3. `generate_angles.py` — Generate Orientations & Strains

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

## Quick Start for Your Teammates

### Step 1: View Existing Data

```bash
pip install -r requirements.txt
python visualize.py ~/EMsoftData/Fe_FCC_exp/Fe_EBSD_patterns.h5
```

This opens:
- A grid view of all patterns
- Intensity profile plots
- Mean/std sensitivity heatmap

### Step 2: Load Data for ML

```python
import numpy as np
from load_data import EBSDDataLoader

loader = EBSDDataLoader('~/EMsoftData/Fe_FCC_exp/Fe_EBSD_patterns.h5')

X = loader.get_patterns()          # images: (N, 480, 640)
X_norm = loader.get_normalized()    # normalized: (N, 480, 640)
y_euler = loader.get_euler()      # orientation: (N, 3)
y_quat  = loader.get_quaternions() # quaternion: (N, 4)
y_strain = loader.get_voigt_strain() # strain: (N, 6) or None

# Ready for PyTorch/TensorFlow:
# X_norm shape: (N, 1, 480, 640)  ← add channel dim for CNN
X_cnn = X_norm[:, np.newaxis, :, :]   # (N, 1, 480, 640)
```

### Step 3: Generate More Data

```bash
# Inside Docker container (after pip install):
python generate_angles.py -n 10000 -s 0.02 --strain-type multiaxial --seed 42

# This creates: Fe_FCC_exp/010000_strain_multiaxial.txt
# Then re-run EMEBSD to generate the patterns:
EMEBSD /home/EMuser/EMPlay/Fe_FCC_exp/EMEBSD.nml
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
