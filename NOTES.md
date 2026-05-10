# Operational Reference — pyEBSDtools

Everything you need to run this codebase: commands, what they do internally,
what files they produce, and how each stage connects to the next.

---

## Stage map

| Stage | Model | Input | Output | Script |
|-------|-------|-------|--------|--------|
| 1 | `SinglePatternModel` | 1 pattern `(B,1,H,W)` | ε `(B,6)` | `train_encoder.py` |
| 2 | `GridModel` | 3×3 grid `(B,9,1,H,W)` | ε at center `(B,6)` | `train_grid.py` |
| 3 | `PairModel` | 2 grids × `(B,9,1,H,W)` | Δε `(B,6)` | `train_pair.py` |

Stage 1 is a debug baseline — it collapses to near-zero strain because the
single-pattern → absolute-strain mapping is ill-posed. Stage 2 adds spatial
context from neighbours. Stage 3 predicts **Δε = ε_B − ε_A** (relative strain
between adjacent scan points), which is well-posed and mirrors HR-EBSD physics.

---

## Quick start

```bash
# Install deps
pip install -r requirements.txt

# Generate Stage 2/3 data (10 000 patterns, 100×100 spatial field, ~2–3 h GPU)
make generate CONFIG=datagen/configs/spatial_100x100.yaml

# Train Stage 3 pair model
python scripts/train_pair.py data.grid_rows=100 data.grid_cols=100

# Watch training live
tensorboard --logdir outputs --port 6006
```

---

## 1. Data generation pipeline

### 1.1 Four stages inside `make generate`

```
config.yaml
  │
  ├─ Stage 1 (make sample)   → datagen/sampler.py + angle_generation.py
  │   Writes to ~/EMsoftData/{experiment_name}/:
  │     {exp}_angles.txt       — EMEBSD input (orpcdef: Euler + F-tensor per pattern)
  │     {exp}_Ftensors.npy     — (N, 3, 3) deformation tensors
  │     {exp}_euler.npy        — (N, 3) Euler angles in degrees
  │     {exp}_positions.npy    — (N, 2) [row, col] per pattern  ← spatial mode only
  │     {exp}_metadata.json    — grid dims + generation settings ← spatial mode only
  │     config_snapshot.yaml   — exact config used
  │
  ├─ Stage 2 (make simulate)  → datagen/emsoft.py → Docker (EMMCOpenCL → EMEBSDmaster → EMEBSD)
  │   Writes to ~/EMsoftData/{experiment_name}/:
  │     Fe_MCoutput.h5         — Monte Carlo + master pattern
  │     Fe_EBSD_patterns.h5    — synthetic pattern stack
  │
  ├─ Stage 3 (make convert)   → datagen/convert.py
  │   Writes to data/processed/ (paths.processed_dir):
  │     X_patterns.npy         — (N, 1, H, W) float32   ← channel-first, PyTorch-ready
  │     y_strain.npy           — (N, 6) float64   Voigt [ε11 ε22 ε33 ε23 ε13 ε12]
  │     y_quaternion.npy       — (N, 4) float64   unit quaternion [q0 q1 q2 q3]
  │     y_euler.npy            — (N, 3) float64   Euler angles (reference)
  │     y_positions.npy        — (N, 2) int32     [row, col]  ← spatial mode only
  │
  └─ Stage 4 (make validate)  → helpers/validate.py
      Prints a sanity-check report (shapes, dtypes, NaN, quaternion norms, strain bounds)
```

> **Only Stage 2 takes a long time** (~2–3 h for 10 000 patterns on GPU).
> All other stages finish in seconds to a few minutes.
> Stages can be run individually — see §1.3.

### 1.2 Which config to use

| Use case | Config |
|----------|--------|
| Stage 2/3 training data (recommended) | `datagen/configs/spatial_100x100.yaml` |
| Stage 1 baseline / custom experiments | `datagen/configs/config.yaml` (edit inline) |

**`spatial_100x100.yaml`** — pre-configured for a 100×100 = 10 000 pattern spatial
field with `field_type: combined`, `seed: 42`, orientation drift ~1°.
Output goes to `~/EMsoftData/Fe_FCC_spatial_100x100/` and `data/spatial_100x100/`.

```bash
# Stage 2/3 data — full run
make generate CONFIG=datagen/configs/spatial_100x100.yaml

# Stage 1 data — edit datagen/configs/config.yaml first, then:
make generate
```

### 1.3 Partial pipeline runs

```bash
# Run individual stages only:
make sample    CONFIG=...   # Stage 1: angles + labels (fast)
make simulate  CONFIG=...   # Stage 2: Docker/EMsoft only (slow)
make convert   CONFIG=...   # Stage 3: HDF5 → .npy (fast)
make validate  CONFIG=...   # Stage 4: sanity check

# Skip specific stages:
make skip-simulate CONFIG=...   # Stages 1, 3, 4 — reuse existing .h5
make skip-sample   CONFIG=...   # Stages 2, 3, 4 — reuse existing angles

# Background (log to file):
make generate-bg CONFIG=datagen/configs/spatial_100x100.yaml
```

### 1.4 Strain field types (spatial mode)

What `SpatialFieldGenerator` actually computes:

| `field_type` | Physical scenario | ε range |
|-------------|------------------|---------|
| `uniaxial_gradient` | Tensile specimen; ε11 ramps linearly across scan, Poisson contraction in ε22/ε33 | 5×10⁻⁴ – 3×10⁻³ |
| `pure_bending` | 3-point bend specimen; ε11 varies through height, zero at neutral axis | 1×10⁻³ – 5×10⁻³ |
| `gaussian_inclusion` | Hard carbide / indenter / void; Gaussian strain bump + shear | 2×10⁻³ – 8×10⁻³ |
| `shear_gradient` | Torsion bar / shear-band precursor; ε12 linear or sinusoidal | 5×10⁻⁴ – 4×10⁻³ |
| `biaxial_gradient` | Thin-film thermal mismatch; ε11=ε22 radially from centre | 5×10⁻⁴ – 4×10⁻³ |
| `combined` | Random weighted mix of 2+ of the above **(default, recommended)** | varies |

All fields are geometrically compatible by construction (derived from smooth
analytic displacement fields or superpositions thereof). Poisson's ratio = 0.29
(Fe) is used for transverse contraction in normal-strain fields.

### 1.5 Position ordering

Every pattern `i` maps to:

```
row = i // grid_cols
col = i % grid_cols
```

Row-major C order, same as EMsoft scan order. `y_positions.npy` makes this
explicit — it's required by the Saint-Venant loss in Stage 3 training.

### 1.6 Strain labels — what convention

`y_strain.npy` is **engineering Voigt** `[ε11, ε22, ε33, 2ε23, 2ε13, 2ε12]`
computed via **Green-Lagrange**: `E = ½(FᵀF − I)`, then off-diagonals × 2.

This is the standard engineering/Voigt convention used throughout the codebase.
The F-tensors fed to EMsoft are `F = I + ε` (small-strain approximation) with
tensor shear components in the off-diagonals.

### 1.7 Useful inspection commands

```bash
# Visualise generated patterns from HDF5
python scripts/visualize.py ~/EMsoftData/Fe_FCC_spatial_100x100/Fe_EBSD_patterns.h5 --grid

# Validate processed data
make validate CONFIG=datagen/configs/spatial_100x100.yaml

# Quick numpy shape check
python -c "import numpy as np; d='data/spatial_100x100'; \
  [print(f, np.load(f'{d}/{f}').shape) for f in \
  ['X_patterns.npy','y_strain.npy','y_positions.npy']]"
```

---

## 2. Docker / EMsoft details

The pipeline manages Docker automatically. Below is what happens under the hood
and what to do if it breaks.

### 2.1 What `make simulate` does internally

```
datagen/emsoft.py:NMLWriter   → writes EMMCOpenCL.nml, EMEBSDmaster.nml,
                                  EMEBSDmasterOCL.nml, BetheParameters.nml, EMEBSD.nml
datagen/emsoft.py:DockerRunner:
  Docker run 1: EMMCOpenCL → Fe_MCoutput.h5    (Monte Carlo, GPU, ~10–30 min once per crystal)
                EMEBSDmasterOpenCL              (master pattern, GPU, ~5–20 min once per crystal)
  Host step:    h5py renames EBSDMasterOpenCLNameList → EBSDMasterNameList  (GPU-master patch)
  Docker run 2: EMEBSD → Fe_EBSD_patterns.h5   (pattern generation, ~2–3 h for 10k patterns)
```

> **Monte Carlo + master pattern only need to run once per crystal structure.**
> If `Fe_MCoutput.h5` already exists, EMsoft skips those steps.
> Use `make skip-simulate` to reuse an existing `.h5` without re-running Docker.

### 2.2 One-time setup

```bash
# Pull Docker image (once)
make docker-pull

# Check Docker + GPU are working
make docker-check

# Set up crystal file (copies Ni.xtal → Fe_FCC.xtal from inside the Docker image)
make setup-xtal
make setup-xtal CONFIG=datagen/configs/spatial_100x100.yaml
```

### 2.3 Expected runtimes

| Step | Hardware | Time |
|------|----------|------|
| EMMCOpenCL (2B electrons) | GPU | ~20–40 min |
| EMEBSDmasterOpenCL | GPU | ~5–20 min |
| EMEBSD — 1 000 patterns | CPU 16 threads | ~15 min |
| EMEBSD — 10 000 patterns | CPU 16 threads | ~2–3 h |
| EMEBSD — 100 000 patterns | CPU 16 threads | ~15–30 h |

---

## 3. Training

> **The Makefile has no training targets.** It is purely for data generation.
> Training and inference are always run directly as `python scripts/...`.

All training scripts use **Hydra** for config. Override any field on the CLI
without `--` prefix: `key=value` or `section.key=value`.

Outputs go to `outputs/YYYY-MM-DD/HH-MM-SS/` automatically.

### 3.1 Stage 1 — single pattern → ε  (debug baseline)

```bash
python scripts/train_encoder.py
# Override common knobs:
python scripts/train_encoder.py training.epochs=100 training.lr=5e-4
python scripts/train_encoder.py model.feature_dim=256 training.predict_orientation=true
python scripts/train_encoder.py training.loss_fn=mae
```

Config: `configs/encoder.yaml` — `data.path` defaults to `data/processed`.

**What happens inside:**
1. Loads `X_patterns.npy` + `y_strain.npy` (+ `y_quaternion.npy` if `predict_orientation=true`)
2. Splits by random index (train/val/test)
3. Normalises input (minmax or zscore) on train split; same stats applied to val
4. Normalises targets (z-score on train Δε)
5. ResNet18 (grayscale-adapted) → 128-d feature → MLP heads → ε + optionally q
6. Loss: Huber on ε + geodesic loss on q (if orientation enabled)
7. Cosine-annealing LR schedule

**Why it fails:** Predicting absolute ε from one pattern is ill-posed — many
orientations/strains produce visually similar patterns. The model collapses to
predicting near the training mean.

---

### 3.2 Stage 2 — 3×3 grid → ε at center

```bash
python scripts/train_grid.py data.grid_rows=100 data.grid_cols=100
# With overrides:
python scripts/train_grid.py data.grid_rows=100 data.grid_cols=100 \
    model.feature_dim=256 training.epochs=100 training.batch_size=8
```

Config: `configs/grid.yaml` — `data.path` defaults to `data/processed`.

**What happens inside:**
1. Loads `X_patterns.npy` + `y_strain.npy`
2. `build_grid_samples(X, y, grid_rows, grid_cols)` reshapes the flat scan into
   a 2D grid and extracts all 3×3 neighbourhoods around interior points
   → `(M, 9, H, W)` grids + `(M, 6)` labels (ε at center, index 4)
   → M = (rows-2) × (cols-2) = 9 604 for a 100×100 scan
3. Random index split, then train/val/test normalisation
4. `GridModel`: shared ResNet18 encodes all 9 patterns → `(B, 128, 3, 3)` feature map
   → `SpatialStrainHead` (Conv2D + center-skip + MLP) → ε(center)

---

### 3.3 Stage 3 — pair of grids → Δε  ← primary model

```bash
python scripts/train_pair.py data.grid_rows=100 data.grid_cols=100
# Common overrides:
python scripts/train_pair.py data.grid_rows=100 data.grid_cols=100 \
    training.epochs=100 training.batch_size=4 training.sv_weight=0.2
# Horizontal pairs only (faster, half the data):
python scripts/train_pair.py data.grid_rows=100 data.grid_cols=100 \
    data.directions=[horizontal]
```

Config: `configs/pair.yaml` — `data.path` defaults to `data/processed`.

**What happens inside:**
1. Loads `X_patterns.npy` + `y_strain.npy`
2. `build_pair_samples(X, y, grid_rows, grid_cols, directions)` produces:
   - `grids_a` `(M, 9, H, W)` — 3×3 neighbourhood around point A
   - `grids_b` `(M, 9, H, W)` — 3×3 neighbourhood around adjacent point B
   - `delta_strain` `(M, 6)` — Δε = ε_B − ε_A
   - `pos_a`, `pos_b` `(M, 2)` — `[row, col]` of each center (used by SV loss)
   - For a 100×100 scan, both directions → ~19 600 pairs
3. Random index split on pairs (note: adjacent pairs overlap in patterns — see §5)
4. `GridPairDataset` returns 5-tuple: `(grid_a, grid_b, targets, pos_a, pos_b)`
5. `PairModel`: same shared encoder for all 18 patterns →
   `F_A (B,128,3,3)`, `F_B (B,128,3,3)` → subtract → `RelativeStrainHead` → Δε

**Loss function (3 terms):**

```
L = L_regression + sv_weight * L_sv + bounds_weight * L_bounds
```

| Term | What it does |
|------|-------------|
| `L_regression` | Huber/MAE/MSE on Δε vs ground truth |
| `L_sv` (Saint-Venant) | Loop consistency: for every complete rectangle in the batch, the sum of Δε around the loop must be zero — i.e., `Δε_h(r,c) + Δε_v(r,c+1) − Δε_h(r+1,c) − Δε_v(r,c) = 0`. This enforces that predictions are integrable (compatible strain field). Weight `sv_weight=0.1` by default. |
| `L_bounds` | Quadratic penalty for any Δε component exceeding `max_abs_strain` (default 5%). Prevents the network from predicting physically impossible large values to cheat regression loss. |

Set `sv_weight=0.0` to run without the SV constraint (pure regression).

**TensorBoard scalars logged:**
- `Loss/train`, `Loss/val` — total loss
- `Loss_SV/train`, `Loss_SV/val` — Saint-Venant term alone
- `Loss_Bounds/train`, `Loss_Bounds/val` — bounds penalty alone
- `DeltaStrainMAE/train`, `DeltaStrainMAE/val`
- `DeltaStrainRMSE/train`, `DeltaStrainRMSE/val`
- `PerComponentMAE_val/delta_{e11..e12}` — per-component

**Key config knobs for pair.yaml:**

| Key | Default | Notes |
|-----|---------|-------|
| `data.grid_rows`, `data.grid_cols` | 100, 100 | Must satisfy rows×cols == N patterns |
| `data.directions` | `[horizontal, vertical]` | Which adjacency directions to pair |
| `model.feature_dim` | 128 | Shared encoder output size; 18× memory vs Stage 1 per batch |
| `training.batch_size` | 8 | Reduce to 4 if OOM |
| `training.sv_weight` | 0.1 | Saint-Venant loop-consistency weight |
| `training.bounds_weight` | 0.01 | Physical-magnitude penalty weight |
| `training.max_abs_strain` | 0.05 | Soft cap on Δε components (5%) |
| `training.loss_fn` | `huber` | `huber \| mae \| mse` |

---

### 3.4 Per-run outputs (all stages)

```
outputs/YYYY-MM-DD/HH-MM-SS/
  checkpoints/
    best.pt                  ← lowest val-loss checkpoint
    last.pt                  ← end-of-training checkpoint
    norm_stats.json          ← input + target normalisation stats
    split_indices.json       ← train/val/test index arrays
  config_snapshot.json       ← exact Hydra config used
  metrics.csv                ← per-epoch: loss, strain_mae, strain_rmse, per-component MAE
  metrics.json               ← same, JSON format
  tensorboard/               ← TensorBoard event files
```

Load norm stats at inference time:
```python
import json, numpy as np
ns = json.load(open("outputs/.../checkpoints/norm_stats.json"))
# Pattern normalisation: ns["min"], ns["max"]  (minmax mode)
# Target denormalisation: ns["y_mean"], ns["y_std"]
pred_physical = pred_normalised * np.array(ns["y_std"]) + np.array(ns["y_mean"])
```

---

## 4. Evaluation and inference (Stage 1)

> Stage 2 and Stage 3 evaluation scripts are not yet written.

### 4.1 Full evaluation — `infer_eval.py`

```bash
# Val split of the most recent run (default):
python scripts/infer_eval.py

# Specific run / split:
python scripts/infer_eval.py --run-dir outputs/2026-04-29/14-00-58 --split val
python scripts/infer_eval.py --split train    # overfit check
python scripts/infer_eval.py --split test     # held-out (only if test_split > 0)
python scripts/infer_eval.py --split all      # full dataset

# Speed / memory control:
python scripts/infer_eval.py --batch-size 128 --max-samples 2000

# Skip scatter plot (headless servers):
python scripts/infer_eval.py --no-plot
```

Outputs (written to run dir):
- `eval_scatter_{split}.png` — 2×3 grid of pred-vs-true scatter for each Voigt component
- `eval_results_{split}.json` — overall MAE, RMSE, max error, per-component MAE

### 4.2 Single-sample inference — `infer.py`

```bash
python scripts/infer.py                        # latest run, sample 0
python scripts/infer.py --index 42             # specific sample
python scripts/infer.py --save-plot pred.png   # save figure
python scripts/infer.py --run-dir outputs/... --checkpoint last.pt
```

---

## 5. Known limitations and caveats

### Train/val split has spatial overlap for Stage 2/3

`build_pair_samples` splits randomly over pairs. Adjacent pairs share 6 of 9
patterns in their grids, so training and validation sets are not independent.
This inflates val metrics slightly. For rigorous evaluation, split by scan
region (e.g., left half for train, right half for val) — not yet implemented.

### Stage 1 collapses

A single EBSD pattern does not contain enough information to uniquely determine
small elastic strains. The model collapses to predicting near-zero strain
(close to the training mean). Use Stage 2 or Stage 3.

### Stage 3 outputs Δε, not absolute ε

To recover the full strain map: accumulate Δε predictions along rows and
columns (cumulative sum or least-squares path integration from a reference
point). This reconstruction script does not exist yet.

### Patterns must stay float32

`X_patterns.npy` stores raw EMsoft float32 intensities. Never convert to PNG
for training — PNG is 8-bit and destroys the subtle intensity gradients that
encode strain information. PNG is for visualisation only.

### SV loop loss activates only when complete rectangles are in the batch

The Saint-Venant loop-consistency loss (`L_sv`) finds rectangular loops whose
four constituent pairs are all in the current batch. With small batch sizes
(e.g., 4 or 8) and random shuffling, only a fraction of batches will contain
complete loops. Increase `batch_size` or temporarily disable shuffling to
increase loop-hit rate. The loss returns 0.0 silently when no loops are found.

---

## 6. Makefile target reference

```
make help                 Print this summary
make venv                 Create .venv and install Python deps
make docker-pull          Pull EMsoft image from Docker Hub
make docker-build         Build EMsoft image locally
make docker-check         Verify Docker + GPU are available
make setup-xtal           Copy Ni.xtal → Fe_FCC.xtal from image

make generate             Full pipeline (stages 1–4)
make generate-bg          Full pipeline in background (log to file)
make sample               Stage 1 only (angles + label .npy files)
make simulate             Stage 2 only (Docker/EMsoft)
make convert              Stage 3 only (HDF5 → .npy)
make validate             Stage 4 only (sanity check)
make skip-simulate        Stages 1, 3, 4 — reuse existing .h5
make skip-sample          Stages 2, 3, 4 — reuse existing angles
make preview              Visualise HDF5 patterns (opens window)

make sync                 rsync code to REMOTE_HOST:REMOTE_DIR
make clean                Remove all generated data
make clean-raw            Remove data/raw/ only
make clean-processed      Remove data/processed/ only

# Pass a different config:
make generate CONFIG=datagen/configs/spatial_100x100.yaml
```

---

## 7. File layout reference

```
datagen/configs/
  config.yaml                ← default (edit for custom runs)
  spatial_100x100.yaml       ← ready-to-use Stage 2/3 config

configs/
  encoder.yaml               ← Stage 1 hyperparams  data.path=data/processed
  grid.yaml                  ← Stage 2 hyperparams  data.path=data/processed
  pair.yaml                  ← Stage 3 hyperparams  data.path=data/processed

data/processed/              ← training .npy files (after make convert)
  X_patterns.npy   (N,1,H,W) float32
  y_strain.npy     (N,6)     float64  Voigt ε
  y_quaternion.npy (N,4)     float64  unit quaternion
  y_euler.npy      (N,3)     float64  Euler degrees
  y_positions.npy  (N,2)     int32    [row,col]  ← spatial mode only

~/EMsoftData/{experiment_name}/   ← raw EMsoft outputs
  {exp}_angles.txt
  {exp}_Ftensors.npy
  {exp}_euler.npy
  {exp}_positions.npy        ← spatial mode only
  {exp}_metadata.json        ← spatial mode only
  config_snapshot.yaml
  Fe_MCoutput.h5
  Fe_EBSD_patterns.h5

outputs/YYYY-MM-DD/HH-MM-SS/     ← per-run training outputs
  checkpoints/best.pt
  checkpoints/last.pt
  checkpoints/norm_stats.json
  checkpoints/split_indices.json
  config_snapshot.json
  metrics.csv / metrics.json
  tensorboard/
```
