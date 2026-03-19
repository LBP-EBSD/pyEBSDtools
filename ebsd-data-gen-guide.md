# EMsoft EBSD Synthetic Data Generation Guide

## Overview

The EBSD pattern generation pipeline has **3 steps**:

```
1. EMMCOpenCL  →  MCoutput.h5     (Monte Carlo electron trajectories)
2. EMEBSDmaster →  EBSDmaster.h5  (Master diffraction pattern)
   └─ Option: EMEBSDmasterOpenCL for GPU acceleration (3-10× faster)
3. EMEBSD       →  patterns.h5    (Synthetic pattern stack)
```

**Note on Fe FCC**: Fe is BCC at room temperature but FCC at high temperature.
The crystal structure files in EMsoft use the `.xtal` (HDF5) format. You can:
- Use `Ni.xtal` (FCC, same structure) as a proxy for Fe FCC
- Or download EMsoftData which has more crystal files

---

## Step 0: Data Persistence Setup

Create a local folder to store all your data so it's NOT lost when the container is removed:

```bash
mkdir -p ~/EMsoftData/Fe_FCC
```

### Run Docker with persistent volume:

```bash
docker run --gpus all \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  -v ~/EMsoftData:/home/EMuser/EMPlay \
  -v ~/EMsoftData/Fe_FCC:/home/EMuser/XtalFolder \
  -it marcdegraef/emsoft_sdk:buildx-latest bash
```

**What this does:**
- `-v ~/EMsoftData:/home/EMuser/EMPlay` → all output data goes to your local `~/EMsoftData`
- `-v ~/EMsoftData/Fe_FCC:/home/EMuser/XtalFolder` → crystal files accessible inside container
- `--gpus all` + nvidia devices → GPU acceleration for EMMCOpenCL

---

## Step 1: Get a Crystal File (inside Docker)

```bash
# Check available crystal files in EMsoftData
ls /home/EMs/EMsoftData/

# Copy Ni (FCC) as a proxy for Fe FCC
cp /home/EMs/EMsoftData/Ni.xtal /home/EMuser/XtalFolder/Fe_FCC.xtal
```

If `Ni.xtal` is not available, clone EMsoftData:

```bash
cd /home/EMs
git clone https://github.com/EMsoft-org/EMsoftData.git
cp /home/EMs/EMsoftData/Ni.xtal /home/EMuser/XtalFolder/Fe_FCC.xtal
```

---

## Step 2: Configure EMsoftConfig.json

The Docker container has a default config. Update paths to match your mounted volumes:

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

---

## Step 3: Create NML Input Files

Create a working directory for your experiment:

```bash
mkdir -p /home/EMuser/EMPlay/Fe_FCC_exp
```

### 3a. Monte Carlo (EMMCOpenCL.nml)

```bash
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
```

### 3b. Master Pattern (EMEBSDmaster.nml)

**Option A — CPU (EMEBSDmaster)** — slower but works on any machine:

```bash
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
```

**Option B — GPU (EMEBSDmasterOpenCL)** — 3–10× faster on NVIDIA/AMD GPU:

```bash
cat > /home/EMuser/EMPlay/Fe_FCC_exp/EMEBSDmasterOCL.nml << 'EOF'
 &EBSDmastervars
  dmin = 0.05,
  npx = 500,
  nthreads = 7,
  platid = 1,
  devid = 1,
  globalworkgrpsz = 150,
  blocksize = 32,
  energyfile = 'Fe_FCC_exp/Fe_MCoutput.h5',
  BetheParametersFile = 'Fe_FCC_exp/BetheParameters.nml',
  restart = .FALSE.,
  uniform = .FALSE.,
 /
EOF
```
> **Note:** `nthreads` must be of the form `4N+3` (minimum 7). Output is identical to CPU version.

### 3c. Bethe Parameters (BetheParameters.nml)

```bash
cat > /home/EMuser/EMPlay/Fe_FCC_exp/BetheParameters.nml << 'EOF'
 &BetheList
  c1 = 8.0,
  c2 = 50.0,
  c3 = 100.0,
  sgdbdiff = 1.0,
 /
EOF
```

### 3d. Pattern Generation with Elastic Strain (EMEBSD.nml)

**Key parameters for elastic strain:**
- `applyDeformation = 'y'` → enables strain
- `Ftensor` → 3x3 deformation gradient tensor F = I + ε (Voigt notation)
- For FCC Fe, `anglefiletype = 'orpcdef'` lets you mix orientations + strains per pattern

```bash
cat > /home/EMuser/EMPlay/Fe_FCC_exp/EMEBSD.nml << 'EOF'
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
  anglefile = 'Fe_FCC_exp/Fe_angles_strains.txt',
  anglefiletype = 'orpcdef',
  eulerconvention = 'tsl',
  masterfile = 'Fe_FCC_exp/Fe_EBSDmaster.h5',
  datafile = 'Fe_FCC_exp/Fe_EBSD_patterns.h5',
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
```

### 3e. Generate Orientation + Strain Input File

Each line has: `euler1 euler2 euler3 xpc ypc F11 F21 F31 F12 F22 F32 F13 F23 F33`

- Euler angles in degrees (Bunge convention, 'tsl')
- Pattern center (xpc, ypc) in pixels
- Ftensor in **column-major** order (as specified in EMEBSD.template)

Example: 3 orientations × 3 strain levels = 9 patterns:

```bash
cat > /home/EMuser/EMPlay/Fe_FCC_exp/Fe_angles_strains.txt << 'EOF'
! No strain (reference)
  0.0   0.0   0.0   0.0  0.0   1.0 0.0 0.0   0.0 1.0 0.0   0.0 0.0 1.0
 90.0  45.0  30.0   0.0  0.0   1.0 0.0 0.0   0.0 1.0 0.0   0.0 0.0 1.0
 45.0  90.0  15.0   0.0  0.0   1.0 0.0 0.0   0.0 1.0 0.0   0.0 0.0 1.0
! Light tension strain (εxx = 0.005)
  0.0   0.0   0.0   0.0  0.0   1.005 0.0 0.0   0.0 1.0 0.0   0.0 0.0 1.0
 90.0  45.0  30.0   0.0  0.0   1.005 0.0 0.0   0.0 1.0 0.0   0.0 0.0 1.0
 45.0  90.0  15.0   0.0  0.0   1.005 0.0 0.0   0.0 1.0 0.0   0.0 0.0 1.0
! Moderate tension strain (εxx = 0.01)
  0.0   0.0   0.0   0.0  0.0   1.01 0.0 0.0   0.0 1.0 0.0   0.0 0.0 1.0
 90.0  45.0  30.0   0.0  0.0   1.01 0.0 0.0   0.0 1.0 0.0   0.0 0.0 1.0
 45.0  90.0  15.0   0.0  0.0   1.01 0.0 0.0   0.0 1.0 0.0   0.0 0.0 1.0
EOF
```

For **bulk ML data generation**, generate many patterns programmatically:

```python
import numpy as np

# Generate random orientations and strains
n_patterns = 10000

orientations = np.random.uniform(0, 360, (n_patterns, 3))  # Euler angles in degrees
# Ftensor = I + epsilon (Voigt), here: uniaxial tension along x
eps_xx = np.random.uniform(0, 0.02, n_patterns)  # 0-2% strain

F = np.zeros((n_patterns, 9))
F[:, 0] = 1.0 + eps_xx  # F11
F[:, 4] = 1.0            # F22
F[:, 8] = 1.0            # F33

xpc = np.zeros(n_patterns)
ypc = np.zeros(n_patterns)

with open('Fe_angles_strains.txt', 'w') as f:
    f.write('! Fe FCC synthetic data for ML\n')
    for i in range(n_patterns):
        f.write(f'{orientations[i,0]:10.4f} {orientations[i,1]:10.4f} {orientations[i,2]:10.4f} '
                f'{xpc[i]:6.2f} {ypc[i]:6.2f} '
                f'{F[i,0]:12.8f} {F[i,1]:12.8f} {F[i,2]:12.8f} '
                f'{F[i,3]:12.8f} {F[i,4]:12.8f} {F[i,5]:12.8f} '
                f'{F[i,6]:12.8f} {F[i,7]:12.8f} {F[i,8]:12.8f}\n')
```

---

## Step 4: Run the Pipeline

```bash
cd /home/EMuser

# Step 1: Monte Carlo (GPU)
EMMCOpenCL Fe_FCC_exp/EMMCOpenCL.nml

# Step 2: Master Pattern — choose ONE:

# Option A: CPU (slower, ~5-20 min)
EMEBSDmaster Fe_FCC_exp/EMEBSDmaster.nml

# Option B: GPU (3-10× faster, ~2-5 min on RTX 3050)
EMEBSDmasterOpenCL Fe_FCC_exp/EMEBSDmasterOCL.nml

# Step 3: Pattern Generation (CPU, with strain)
EMEBSD Fe_FCC_exp/EMEBSD.nml
```

**Expected runtime:**
- EMMCOpenCL: ~10-30 min (GPU)
- EMEBSDmaster: ~5-20 min (CPU) | EMEBSDmasterOpenCL: ~2-5 min (GPU, RTX 3050)
- EMEBSD: ~1-5 min per 100 patterns (CPU)

---

## Step 5: Extract Data from Container

Since your data is mounted to `~/EMsoftData`, it's already on your host:

```bash
ls ~/EMsoftData/Fe_FCC_exp/
# Fe_MCoutput.h5    (Monte Carlo)
# Fe_EBSDmaster.h5  (Master pattern)
# Fe_EBSD_patterns.h5  ← This is your ML training data!
```

### Reading the HDF5 output (Python):

```python
import h5py
import numpy as np

with h5py.File('~/EMsoftData/Fe_FCC_exp/Fe_EBSD_patterns.h5', 'r') as f:
    print(list(f.keys()))
    # Patterns are typically stored as:
    # patterns = f['EBSD/Data/patterns'][:]
    # Shape: (num_patterns, numsy, numsx) → (N, 480, 640)
    print(patterns.shape)
```

---

## Quick Reference: Ftensor for Elastic Strain

The deformation tensor `F = I + ε` where ε is the elastic strain tensor.

| Strain Type | Ftensor (column-major) |
|---|---|
| No strain | `1 0 0 0 1 0 0 0 1` |
| Tension εxx=0.01 | `1.01 0 0 0 1 0 0 0 1` |
| Tension εxx=0.02 | `1.02 0 0 0 1 0 0 0 1` |
| Compression εxx=-0.01 | `0.99 0 0 0 1 0 0 0 1` |
| Equibiaxial εxx=εyy=0.01 | `1.01 0 0 0 1.01 0 0 0 1` |
| Shear εxy=0.01 | `1 0.01 0 0 1 0 0 0 1` |

---

## Troubleshooting

**Q: `clGetPlatformIDs` error in EMMCOpenCL**
A: OpenCL not visible inside container. Check GPU passthrough:
```bash
clinfo | grep -i nvidia
```

**Q: Crystal file not found**
A: Crystal files must be in the folder pointed to by `EMXtalFolderpathname` in EMsoftConfig.json. Default: `/home/EMuser/XtalFolder`

**Q: "Permission denied" writing to EMPlay**
A: The mounted volume must be writable. Check:
```bash
ls -la ~/EMsoftData/
chmod 777 ~/EMsoftData
```

---

## Step 6: Using the Data with Python / ML

Python tools are provided in `pyEMsoft/EBSDtools/`:

```
pyEMsoft/EBSDtools/
├── requirements.txt     # pip install -r requirements.txt
├── load_data.py         # Load patterns from HDF5
├── visualize.py         # Visualize patterns
├── generate_angles.py   # Generate bulk orientations + strains
└── README.md            # Full documentation
```

### Install dependencies

```bash
pip install numpy h5py matplotlib Pillow
```

### View your patterns

```bash
python pyEMsoft/EBSDtools/visualize.py ~/EMsoftData/Fe_FCC_exp/Fe_EBSD_patterns.h5
```

### Load data for ML

```python
from pyEMsoft.EBSDtools.load_data import EBSDDataLoader

loader = EBSDDataLoader('~/EMsoftData/Fe_FCC_exp/Fe_EBSD_patterns.h5')

X = loader.get_patterns()              # (N, 480, 640) pattern images
y_euler = loader.get_euler()          # (N, 3) Euler angles [degrees]
y_quat  = loader.get_quaternions()     # (N, 4) quaternions
y_strain = loader.get_voigt_strain()  # (N, 6) Voigt strain or None

# Normalize for CNN input
X_norm = loader.get_normalized('minmax')  # 0-1 range
X_cnn = X_norm[:, np.newaxis, :, :]       # (N, 1, 480, 640)

# Dump to .npy files
loader.to_numpy('~/EMsoftData/Fe_FCC_exp/')
```

### Generate bulk orientation/strain files

Run this **on your host** (not in Docker) or inside Docker:

```bash
# 10,000 patterns with random multiaxial strain (0-2%)
python pyEMsoft/EBSDtools/generate_angles.py \
    -n 10000 -s 0.02 --strain-type multiaxial --seed 42

# Then copy the output .txt to the Docker folder
cp ~/EMsoftData/Fe_FCC_exp/010000_strain_multiaxial.txt \
   ~/EMsoftData/Fe_FCC_exp/my_angles.txt

# Inside Docker, update EMEBSD.nml to point to the new file:
#   anglefile = 'Fe_FCC_exp/my_angles.txt'
# Then re-run:
EMEBSD /home/EMuser/EMPlay/Fe_FCC_exp/EMEBSD.nml
```

See `pyEMsoft/EBSDtools/README.md` for full documentation.

---

## What Is an EBSD Pattern?

Each "pattern" is a **480×640 pixel grayscale image** — a simulated photograph of
electron diffraction from a crystal. The bright bands (Kikuchi lines) encode:

- **Crystal orientation** — where the crystal axes point
- **Elastic strain** — how the lattice is stretched

Think of it as a "fingerprint" of the crystal at that orientation and strain state.
For ML: the image is **X** (input), the Euler angles/strain are **y** (labels).

