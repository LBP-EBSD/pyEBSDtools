# Model Usage Notes (Current)

This file is the quick operational reference for data prep, training, evaluation,
and inference in this repo.

---

## 0) Three-stage model overview

| Stage | Model | Input | Output | Script |
|-------|-------|-------|--------|--------|
| 1 | `SinglePatternModel` | 1 pattern `(B,1,H,W)` | ε `(B,6)` | `train_encoder.py` |
| 2 | `GridModel` | 3×3 grid `(B,9,1,H,W)` | ε at center `(B,6)` | `train_grid.py` |
| 3 | `PairModel` | 2 grids `(B,9,1,H,W)` × 2 | Δε `(B,6)` | `train_pair.py` |

Stage 1 is a debug baseline — it collapses to mean-zero strain because the mapping
is ill-posed. Stage 2 adds spatial context. Stage 3 predicts **relative** strain
(Δε = ε_B − ε_A), which is the physically meaningful, well-posed target.

Weights from every run are saved to `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/`.

---

## 1) Setup

From repo root:

```bash
pip install -r requirements.txt
```

Optional TensorBoard:

```bash
tensorboard --logdir outputs --port 6006
```

---

## 2) Data preparation workflow

### 2.1 Generate EMsoft angle/strain file

Script: `scripts/generate_angles.py`

Two modes: **random** (Stage 1 baseline) and **spatial** (Stage 2 / 3 training data).

#### Random mode — Stage 1 baseline

```bash
python scripts/generate_angles.py -n 10000 --strain-type multiaxial -s 0.02 --seed 42
python scripts/generate_angles.py -n 1000 --orient-only
```

Main flags:
- `-n, --n-patterns`
- `-o, --output`, `--seed`, `--xpc`, `--ypc`, `--L`, `--comment`
- `--strain-type` (`uniform|uniaxial_x|uniaxial_y|uniaxial_z|biaxial_xy|biaxial_yz|biaxial_xz|multiaxial|shear_xy|shear_xz|shear_yz|random|none`)
- `-s, --strain-magnitude`   max strain e.g. `0.02` = 2%
- `--uniform-strain`         fixed hydrostatic strain
- `--orient-only`            no strain tensor

---

#### Spatial mode — Stage 2 / 3 training data

Generates a **spatially correlated, physically realistic strain field** on a 2D scan
grid. Pattern `i` maps to `(i // grid_cols, i % grid_cols)` (row-major).

> `grid_rows × grid_cols` must equal the number of patterns EMsoft will produce.
> Set `ipf_wd = grid_cols` and `ipf_ht = grid_rows` in `EMEBSD.nml`.

```bash
# Recommended starting point — 100×100 scan, combined field, orientation drift on
python scripts/generate_angles.py --spatial-field \
    --grid-rows 100 --grid-cols 100 --seed 42

# Pure bending, constant orientation (isolates strain signal)
python scripts/generate_angles.py --spatial-field \
    --grid-rows 100 --grid-cols 100 \
    --field-type pure_bending --constant-orientation --seed 42

# Gaussian inclusion / stress concentration, 2× larger strain
python scripts/generate_angles.py --spatial-field \
    --grid-rows 64 --grid-cols 64 \
    --field-type gaussian_inclusion --field-scale 2.0 --seed 7

# Shear gradient, perfectly smooth (no noise)
python scripts/generate_angles.py --spatial-field \
    --grid-rows 100 --grid-cols 100 \
    --field-type shear_gradient --noise-frac 0 --seed 1

# Uniaxial tension, 2° misorientation drift across scan (realistic subgrain)
python scripts/generate_angles.py --spatial-field \
    --grid-rows 100 --grid-cols 100 \
    --field-type uniaxial_gradient --orientation-spread 2.0 --seed 3
```

**Field types (`--field-type`):**

| Type | Physical case | ε magnitude |
|------|--------------|-------------|
| `uniaxial_gradient` | Tensile specimen along loading axis | 5e-4 – 3e-3 |
| `pure_bending` | 3-pt bending through cross-section height | 1e-3 – 5e-3 |
| `gaussian_inclusion` | Carbide / indentation / void concentration | 2e-3 – 8e-3 |
| `shear_gradient` | Torsion bar / shear-band precursor | 5e-4 – 4e-3 |
| `biaxial_gradient` | Thin-film thermal mismatch / pressure field | 5e-4 – 4e-3 |
| `combined` | Random weighted mix of the above **(recommended)** | varies |

**Spatial mode flags:**

- `--spatial-field`           activate spatial mode (required)
- `--grid-rows / --grid-cols` scan dimensions (default 100×100)
- `--field-type`              see table above (default `combined`)
- `--field-scale`             multiplier on strain (default 1.0; try 0.5–2.0)
- `--constant-orientation`    single fixed orientation for all points (default: drifting)
- `--orientation-spread`      RMS misorientation drift in degrees (default 1.0°; realistic: 0.5–2°)
- `--noise-frac`              spatially-correlated noise as fraction of peak strain (default 0.05)

Output file (auto): `data/raw/{rows}x{cols}_{field_type}_spatial.txt`

The script prints per-component strain statistics and the exact `EMEBSD.nml` snippet.

### 2.2 Run EMsoft (Docker) — generate the HDF5 file

> This is the **only step that takes hours**. Everything else is fast.
> Steps 2.1 and 2.3–2.5 all run on the host (no Docker needed).

EMsoft has 3 sub-steps. Steps A and B only need to run **once per crystal structure**.
Only Step C (EMEBSD) needs to re-run each time you change the angle/strain file.

#### One-time setup: start Docker with persistent storage

```bash
mkdir -p ~/EMsoftData/Fe_FCC_exp

docker run --gpus all \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  -v ~/EMsoftData:/home/EMuser/EMPlay \
  -v ~/EMsoftData/Fe_FCC:/home/EMuser/XtalFolder \
  -it marcdegraef/emsoft_sdk:buildx-latest bash
```

Everything written inside `/home/EMuser/EMPlay/` appears on your host at `~/EMsoftData/`.

#### One-time setup: crystal file (inside Docker)

```bash
# Copy Ni.xtal (FCC proxy for Fe FCC) — only needed once
ls /home/EMs/EMsoftData/                        # check what's available
cp /home/EMs/EMsoftData/Ni.xtal /home/EMuser/XtalFolder/Fe_FCC.xtal
# If not found, clone EMsoftData first:
# cd /home/EMs && git clone https://github.com/EMsoft-org/EMsoftData.git
```

#### One-time setup: EMsoftConfig.json (inside Docker, first time only)

```bash
cat > ~/.config/EMsoft/EMsoftConfig.json << 'EOF'
{
    "EMsoftpathname": "/home/EMs/EMsoft/",
    "EMXtalFolderpathname": "/home/EMuser/XtalFolder",
    "EMdatapathname": "/home/EMuser/EMPlay",
    "EMtmppathname": "/home/EMuser/.config/EMsoft/tmp/",
    "EMsoftLibraryLocation": "/home/EMs/EMsoftBuild/Release/Bin/",
    "EMNotify": "",
    "Develop": "No",
    "UserName": "DockerUser",
    "UserLocation": "DockerContainer",
    "Release": "Yes"
}
EOF
```

#### Step A — Monte Carlo (run once per crystal, ~10–30 min GPU)

```bash
# Inside Docker:
cat > /home/EMuser/EMPlay/Fe_FCC_exp/EMMCOpenCL.nml << 'EOF'
 &MCCLdata
  mode = 'full',
  xtalname = 'Fe_FCC.xtal',
  sig = 70.0,
  omega = 0.0,
  numsx = 801,
  num_el = 10,
  globalworkgrpsz = 150,
  totnum_el = 2000000000,
  multiplier = 1,
  EkeV = 30.0,
  Ehistmin = 15.0,
  Ebinsize = 1.0,
  depthmax = 100.0,
  depthstep = 1.0,
  platid = 1,
  devid = 1,
  dataname = 'Fe_FCC_exp/Fe_MCoutput.h5',
  Notify = 'Off',
 /
EOF

EMMCOpenCL /home/EMuser/EMPlay/Fe_FCC_exp/EMMCOpenCL.nml
```

#### Step B — Master pattern (run once per crystal, ~2–20 min)

```bash
# Inside Docker:
cat > /home/EMuser/EMPlay/Fe_FCC_exp/EMEBSDmaster.nml << 'EOF'
 &EBSDmastervars
  dmin = 0.05,
  npx = 500,
  nthreads = 4,
  doLegendre = .FALSE.,
  energyfile = 'Fe_FCC_exp/Fe_MCoutput.h5',
  BetheParametersFile = 'Fe_FCC_exp/BetheParameters.nml',
  Notify = 'Off',
 /
EOF

cat > /home/EMuser/EMPlay/Fe_FCC_exp/BetheParameters.nml << 'EOF'
 &BetheList
  c1 = 8.0,
  c2 = 50.0,
  c3 = 100.0,
  sgdbdiff = 1.0,
 /
EOF

# CPU (slower):
EMEBSDmaster /home/EMuser/EMPlay/Fe_FCC_exp/EMEBSDmaster.nml

# GPU alternative (3–10× faster):
# EMEBSDmasterOpenCL /home/EMuser/EMPlay/Fe_FCC_exp/EMEBSDmasterOCL.nml
```

#### Step C — Pattern generation (re-run for each new dataset)

This is the step you re-run when you have a new angle/strain file.

**For Stage 1 (random patterns):**

```bash
# On host — generate the angle file:
python scripts/generate_angles.py \
    -n 10000 --strain-type multiaxial -s 0.02 --seed 42 \
    -o ~/EMsoftData/Fe_FCC_exp/stage1_10k_multiaxial.txt

# Inside Docker — run EMEBSD:
cat > /home/EMuser/EMPlay/Fe_FCC_exp/EMEBSD_stage1.nml << 'EOF'
 &EBSDdata
  L = 15000.0,
  thetac = 10.0,
  delta = 50.0,
  numsx = 640,
  numsy = 480,
  xpc = 0.0,
  ypc = 0.0,
  energymin = 10.0,
  energymax = 25.0,
  includebackground = 'n',
  anglefile = 'Fe_FCC_exp/stage1_10k_multiaxial.txt',
  anglefiletype = 'orpcdef',
  eulerconvention = 'tsl',
  masterfile = 'Fe_FCC_exp/Fe_EBSDmaster.h5',
  datafile = 'Fe_FCC_exp/stage1_10k_multiaxial.h5',
  bitdepth = 'float',
  beamcurrent = 150.0,
  dwelltime = 100.0,
  poisson = 'n',
  binning = 1,
  applyDeformation = 'y',
  Fframe = 'crys',
  scalingmode = 'not',
  gammavalue = 1.0,
  makedictionary = 'n',
  maskpattern = 'n',
  nthreads = 4,
 /
EOF

EMEBSD /home/EMuser/EMPlay/Fe_FCC_exp/EMEBSD_stage1.nml
# Output: ~/EMsoftData/Fe_FCC_exp/stage1_10k_multiaxial.h5
```

**For Stage 2 / 3 (spatial field):**

```bash
# On host — generate spatial angle file:
python scripts/generate_angles.py \
    --spatial-field --grid-rows 100 --grid-cols 100 \
    --field-type combined --seed 42 \
    -o ~/EMsoftData/Fe_FCC_exp/spatial_100x100_combined.txt
# (script prints the exact ipf_wd/ipf_ht values to put in the nml)

# Inside Docker — run EMEBSD:
cat > /home/EMuser/EMPlay/Fe_FCC_exp/EMEBSD_spatial.nml << 'EOF'
 &EBSDdata
  L = 15000.0,
  thetac = 10.0,
  delta = 50.0,
  numsx = 640,
  numsy = 480,
  xpc = 0.0,
  ypc = 0.0,
  energymin = 10.0,
  energymax = 25.0,
  includebackground = 'n',
  anglefile = 'Fe_FCC_exp/spatial_100x100_combined.txt',
  anglefiletype = 'orpcdef',
  eulerconvention = 'tsl',
  masterfile = 'Fe_FCC_exp/Fe_EBSDmaster.h5',
  datafile = 'Fe_FCC_exp/spatial_100x100_combined.h5',
  bitdepth = 'float',
  beamcurrent = 150.0,
  dwelltime = 100.0,
  poisson = 'n',
  binning = 1,
  applyDeformation = 'y',
  Fframe = 'crys',
  scalingmode = 'not',
  gammavalue = 1.0,
  makedictionary = 'n',
  maskpattern = 'n',
  nthreads = 4,
 /
EOF
# Note: numsx=640, numsy=480 are PATTERN pixel dimensions, not scan grid dims.
# The scan grid (100×100=10000 patterns) comes entirely from the angle file.

EMEBSD /home/EMuser/EMPlay/Fe_FCC_exp/EMEBSD_spatial.nml
# Output: ~/EMsoftData/Fe_FCC_exp/spatial_100x100_combined.h5
```

**Expected runtimes (EMEBSD, CPU, 4 threads):**

| N patterns | Time |
|-----------|------|
| 1 000 | ~15 min |
| 10 000 | ~2–3 h |
| 100 000 | ~15–30 h |

---

### 2.3 Convert HDF5 to NumPy — dataset naming convention

> **Datasets should not change.** EMsoft runs take hours. Each dataset lives in
> its own named subdirectory and is never overwritten accidentally.
> The `--force` flag is required to overwrite existing files.

```bash
# Stage 1 dataset
python scripts/prepare_numpy.py \
    --h5 ~/EMsoftData/Fe_FCC_exp/stage1_10k_multiaxial.h5 \
    --out data/stage1_10k_multiaxial/

# Spatial dataset (Stage 2 / 3)
python scripts/prepare_numpy.py \
    --h5 ~/EMsoftData/Fe_FCC_exp/spatial_100x100_combined.h5 \
    --out data/spatial_100x100_combined/

# If you genuinely need to re-convert (e.g. re-ran EMsoft):
python scripts/prepare_numpy.py \
    --h5 ~/EMsoftData/Fe_FCC_exp/spatial_100x100_combined.h5 \
    --out data/spatial_100x100_combined/ --force
```

Flags:
- `--h5` (required)
- `--out`   — always use a descriptive named subdirectory
- `--force` — required to write into a directory that already has `.npy` files

Output files (per dataset directory):
- `X_patterns.npy`   — `(N, H, W)` float32
- `y_euler.npy`      — `(N, 3)` Bunge Euler in degrees
- `y_quaternion.npy` — `(N, 4)` unit quaternion
- `y_strain.npy`     — `(N, 6)` Voigt strain `[ε11 ε22 ε33 ε23 ε13 ε12]`

### 2.4 Visualize raw patterns

Script: `scripts/visualize.py`

```bash
python scripts/visualize.py /path/to/your_data.h5 --grid --heatmap
```

Main flags:
- positional `data_path`
- `--output-dir, -o`
- `--grid`
- `--single N`
- `--compare N M`
- `--heatmap`
- `--histogram`
- `--export-tiff`
- `--export-png`
- `--all`

---

## 3) Training workflow

### 3.1 Stage 1 — single pattern → ε (debug baseline)

```bash
python scripts/train_encoder.py data.path=data/stage1_10k_multiaxial
python scripts/train_encoder.py data.path=data/stage1_10k_multiaxial training.epochs=100 training.lr=5e-4
python scripts/train_encoder.py data.path=data/stage1_10k_multiaxial model.feature_dim=256
python scripts/train_encoder.py data.path=data/stage1_10k_multiaxial training.predict_orientation=true
```

Config: `configs/encoder.yaml`

Key fields:
- `training.epochs`, `training.batch_size`, `training.lr`
- `training.val_split`, `training.test_split` (set `0.0` to disable)
- `training.norm_method` (`minmax|zscore`)
- `training.loss_fn` (`huber|mae|mse`), `training.huber_delta`
- `training.predict_orientation`, `training.orientation_loss_weight`
- `model.feature_dim`
- `data.path`, `data.patterns_file`, `data.strain_file`, `data.orientation_file`

---

### 3.2 Stage 2 — 3×3 grid → ε at center

```bash
python scripts/train_grid.py \
    data.path=data/spatial_100x100_combined data.grid_rows=100 data.grid_cols=100
python scripts/train_grid.py \
    data.path=data/spatial_100x100_combined data.grid_rows=100 data.grid_cols=100 \
    model.feature_dim=256 training.epochs=100
```

Config: `configs/grid.yaml`

Key fields (on top of the shared ones above):
- `data.grid_rows`, `data.grid_cols` — **required**; must satisfy `rows × cols == N patterns`
- `model.feature_dim` — shared encoder output size (default 128; 9× memory vs Stage 1 per batch)
- `training.batch_size` — default 16; reduce if OOM

How grids are built from the flat scan:
- Interior points only (1 ≤ r ≤ rows-2, 1 ≤ c ≤ cols-2) → M = (rows-2)×(cols-2) samples
- Boundary patterns are discarded (no complete 3×3 ring)

---

### 3.3 Stage 3 — pair of grids → Δε (relative strain)

```bash
python scripts/train_pair.py \
    data.path=data/spatial_100x100_combined data.grid_rows=100 data.grid_cols=100
python scripts/train_pair.py \
    data.path=data/spatial_100x100_combined data.grid_rows=100 data.grid_cols=100 \
    data.directions=[horizontal] training.batch_size=4
```

Config: `configs/pair.yaml`

Key fields:
- `data.grid_rows`, `data.grid_cols` — **required**
- `data.directions` — list of `horizontal` and/or `vertical` (default both)
  - `horizontal`: A=(r,c) → B=(r,c+1)
  - `vertical`:   A=(r,c) → B=(r+1,c)
- `model.feature_dim` — default 128; 18× memory vs Stage 1 per batch
- `training.batch_size` — default 8; reduce if OOM

What the model learns:
- Input: two 3×3 grids around adjacent scan points A and B
- Architecture: shared encoder → F_A, F_B → subtract → conv head → Δε
- Subtraction cancels shared orientation/intensity bias, isolates deformation signal

To reconstruct absolute ε from predictions: accumulate Δε along rows/columns
(cumulative sum or least-squares integration — reconstruction script TBD).

---

### Per-run outputs (all three stages)

Inside `outputs/YYYY-MM-DD/HH-MM-SS/`:
- `checkpoints/best.pt`            — lowest val loss checkpoint
- `checkpoints/last.pt`            — end-of-training checkpoint
- `checkpoints/norm_stats.json`    — input + target normalisation stats
- `checkpoints/split_indices.json` — train/val/test index arrays
- `config_snapshot.json`           — exact config used
- `metrics.csv` / `metrics.json`   — per-epoch metrics
- `tensorboard/`                   — TensorBoard event files

---

## 4) Evaluation workflow (Stage 1 only, split-aware)

Script: `scripts/infer_eval.py`

Default behavior:
- uses latest run
- evaluates on `val` split

```bash
python scripts/infer_eval.py
python scripts/infer_eval.py --run-dir outputs/2026-04-29/14-00-58 --split val
python scripts/infer_eval.py --split train
python scripts/infer_eval.py --split test
python scripts/infer_eval.py --split all
python scripts/infer_eval.py --checkpoint last.pt --batch-size 128 --max-samples 1000
```

Flags:
- `--run-dir` (default latest)
- `--split` (`val|train|test|all`, default `val`)
- `--data-dir` (default from run `config_snapshot.json`)
- `--patterns-file`
- `--strain-file`
- `--checkpoint` (`best.pt|last.pt`)
- `--batch-size`
- `--max-samples`
- `--spot-checks`
- `--no-plot`

Outputs:
- `eval_scatter_<split>.png`
- `eval_results_<split>.json`

> Stage 2 and Stage 3 eval scripts are not yet written.

---

## 5) Single-sample inference (Stage 1)

Script: `scripts/infer.py`

```bash
python scripts/infer.py
python scripts/infer.py --run-dir outputs/2026-04-29/14-00-58 --index 5
python scripts/infer.py --save-plot pred.png
```

Flags:
- `--run-dir` (default latest)
- `--data-dir` (default from run config)
- `--patterns-file`
- `--index`
- `--checkpoint`
- `--save-plot`

Notes:
- Predictions are denormalized back to physical strain units.
- Script is forward-only (expects `y_mean/y_std` in `norm_stats.json`).

---

## 6) Practical defaults and recommendations

- Use `--split val` for model quality checks.
- Use `--split train` only for overfit/debug checks.
- Use `--split test` only when `training.test_split > 0`.
- Keep `--split all` for broad sanity checks, not for unbiased performance.
- If evaluating older runs that predate split index saving, retrain once with
  current `train_encoder.py`.
- For Stage 2 / 3: random split at the sample level has pattern overlap between
  splits (adjacent grids share up to 6 of 9 patterns). Fine for development.
  For rigorous evaluation, split by scan region.
