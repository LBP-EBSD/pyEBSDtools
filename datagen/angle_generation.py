#!/usr/bin/env python3
"""
EMsoft EBSD Angle & Strain Generator
===================================
Generates the input file for EMEBSD pattern generation with random orientations
and/or elastic strains. Produces the .txt file that EMEBSD reads with
`anglefiletype = 'orpcdef'`.

Used by ``datagen.sampler`` (``make generate``) and by the thin CLI wrapper
``../generate_angles.py`` in the EBSDtools directory.

Usage:
    python generate_angles.py                           # 10000 random orientations, no strain
    python generate_angles.py -n 5000 -s 0.01         # 5000 patterns, 0-1% strain
    python generate_angles.py -n 1000 --strain-type uniaxial_x
    python generate_angles.py -n 500 --strain-type multiaxial --uniform-strain 0.005
    python generate_angles.py -n 100 --orient-only     # orientations only, no strain

The output file format (for EMEBSD with anglefiletype='orpcdef'):
    Line 1: 'eu'              ← type tag (Euler angles)
    Line 2: N                 ← number of patterns
    Line 3+: 15 values per line:
        euler1 euler2 euler3 xpc ypc L F11 F21 F31 F12 F22 F32 F13 F23 F33
        (F = deformation tensor = I + strain, column-major order)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# EBSDtools/ — so ``helpers.*`` imports work when this module is ``datagen.angle_generation``
_EBSDTOOLS_ROOT = Path(__file__).resolve().parent.parent
if str(_EBSDTOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EBSDTOOLS_ROOT))

import numpy as np


class AngleStrainGenerator:
    """
    Generates random Euler angles and deformation tensors for EBSD pattern synthesis.

    The deformation tensor F relates to elastic strain via: F = I + ε
    For small strains, the Voigt format is: [ε11, ε22, ε33, ε23, ε13, ε12]

    Strain types supported:
        - uniform:         F = I (no strain) — for pure orientation learning
        - uniaxial_x:      tension/compression along x-axis
        - uniaxial_y:      tension/compression along y-axis
        - uniaxial_z:      tension/compression along z-axis
        - biaxial_xy:      equal tension in x and y
        - biaxial_yz:      equal tension in y and z
        - biaxial_xz:      equal tension in x and z
        - multiaxial:      random strain in all 6 Voigt components
        - shear_xy:        pure shear in xy plane
        - shear_xz:        pure shear in xz plane
    """

    def __init__(self, n_patterns=10000, seed=None):
        self.n_patterns = n_patterns
        self.rng = np.random.default_rng(seed)
        self.strain_type = 'uniform'
        self.strain_magnitude = 0.0
        self.uniform_strain_value = 0.0
        self.xpc = 0.0
        self.ypc = 0.0
        self.L = 15000.0

    def set_strain(self, strain_type, magnitude, uniform_value=0.0):
        self.strain_type = strain_type
        self.strain_magnitude = magnitude
        self.uniform_strain_value = uniform_value

    def set_pattern_center(self, xpc=0.0, ypc=0.0, L=15000.0):
        self.xpc = xpc
        self.ypc = ypc
        self.L = L

    def _identity_F(self):
        return np.eye(3, dtype=np.float64)

    def _strain_to_F(self, eps):
        """
        Convert Voigt-format strain tensor ε to deformation tensor F = I + ε.
        Voigt order: [ε11, ε22, ε33, ε23, ε13, ε12]
        """
        F = np.eye(3, dtype=np.float64)
        F[0, 0] = float(1.0 + eps[0])
        F[1, 1] = float(1.0 + eps[1])
        F[2, 2] = float(1.0 + eps[2])
        F[2, 1] = float(eps[3])
        F[1, 2] = float(eps[3])
        F[2, 0] = float(eps[4])
        F[0, 2] = float(eps[4])
        F[1, 0] = float(eps[5])
        F[0, 1] = float(eps[5])
        return F

    def _random_uniaxial(self, magnitude):
        """Random uniaxial strain along a random axis."""
        axis = self.rng.integers(0, 3)
        eps = np.zeros(6)
        val = self.rng.uniform(-magnitude, magnitude)
        eps[axis] = val
        return self._strain_to_F(eps)

    def _random_biaxial(self, magnitude):
        """Random biaxial strain in a random plane."""
        plane = self.rng.integers(0, 3)
        eps = np.zeros(6)
        val = self.rng.uniform(-magnitude, magnitude)
        if plane == 0:
            eps[0] = val
            eps[1] = val
        elif plane == 1:
            eps[1] = val
            eps[2] = val
        else:
            eps[0] = val
            eps[2] = val
        return self._strain_to_F(eps)

    def _random_multiaxial(self, magnitude):
        """Random strain in all 6 Voigt components."""
        eps = self.rng.uniform(-magnitude, magnitude, size=6)
        return self._strain_to_F(eps)

    def _random_shear(self, magnitude):
        """Random shear strain."""
        plane = self.rng.integers(0, 3)
        eps = np.zeros(6)
        val = self.rng.uniform(-magnitude, magnitude)
        eps[plane + 3] = val
        return self._strain_to_F(eps)

    def _random_uniform_strain(self, value):
        """Uniform (hydrostatic) strain of fixed magnitude."""
        eps = np.array([value, value, value, 0, 0, 0])
        return self._strain_to_F(eps)

    def generate(self, strain_type='uniform', strain_magnitude=0.0, uniform_strain_value=0.0):
        """
        Generate orientations and deformation tensors.

        Args:
            strain_type: 'uniform', 'uniaxial_x/y/z', 'biaxial_xy/yz/xz',
                         'multiaxial', 'shear_xy/xz', 'custom'
            strain_magnitude: maximum absolute strain value (e.g., 0.02 = 2%)
            uniform_strain_value: fixed uniform strain value (for 'uniform' type)

        Returns:
            orientations: (N, 3) Euler angles in degrees [Bunge convention]
            F_tensors: (N, 3, 3) deformation tensors [column-major]
        """
        orientations = self.rng.uniform(0.0, 360.0, size=(self.n_patterns, 3))

        F_tensors = []
        for _ in range(self.n_patterns):
            if strain_type == 'uniform':
                F_tensors.append(self._identity_F())
            elif strain_type in ('uniaxial_x', 'uniaxial_y', 'uniaxial_z'):
                F_tensors.append(self._random_uniaxial(strain_magnitude))
            elif strain_type in ('biaxial_xy', 'biaxial_yz', 'biaxial_xz'):
                F_tensors.append(self._random_biaxial(strain_magnitude))
            elif strain_type == 'multiaxial':
                F_tensors.append(self._random_multiaxial(strain_magnitude))
            elif strain_type in ('shear_xy', 'shear_xz', 'shear_yz'):
                F_tensors.append(self._random_shear(strain_magnitude))
            elif strain_type == 'random':
                choice = self.rng.choice(
                    ['multiaxial', 'uniaxial_x', 'biaxial_xy', 'shear_xy'],
                    p=[0.5, 0.2, 0.15, 0.15]
                )
                if choice == 'multiaxial':
                    F_tensors.append(self._random_multiaxial(strain_magnitude))
                elif choice == 'uniaxial_x':
                    F_tensors.append(self._random_uniaxial(strain_magnitude))
                elif choice == 'biaxial_xy':
                    F_tensors.append(self._random_biaxial(strain_magnitude))
                else:
                    F_tensors.append(self._random_shear(strain_magnitude))
            else:
                F_tensors.append(self._identity_F())

        return orientations, np.array(F_tensors)

    def write_orpcdef(self, orientations, F_tensors, output_path, comment=''):
        """
        Write the orientation + deformation tensor file in EMsoft 'orpcdef' format.

        Format:
            Line 1: 'eu' (Euler angle type)
            Line 2: N (number of patterns)
            Line 3+: one line per pattern (15 reals — no comment lines; EMsoft EBSDreadorpcdef):
                euler1 euler2 euler3 xpc ypc L F11 F21 F31 F12 F22 F32 F13 F23 F33
        """
        N = len(orientations)

        lines = []
        lines.append('eu')
        lines.append(str(N))
        if comment:
            lines.append(f'! {comment}')

        for i in range(N):
            e = orientations[i]
            F = F_tensors[i]
            line = (f'{e[0]:10.4f} {e[1]:10.4f} {e[2]:10.4f} '
                    f'{self.xpc:6.2f} {self.ypc:6.2f} {self.L:12.4f} '
                    f'{F[0,0]:14.8f} {F[1,0]:14.8f} {F[2,0]:14.8f} '
                    f'{F[0,1]:14.8f} {F[1,1]:14.8f} {F[2,1]:14.8f} '
                    f'{F[0,2]:14.8f} {F[1,2]:14.8f} {F[2,2]:14.8f}')
            lines.append(line)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        print(f'[OK] Wrote {N} orientation+strain entries to: {output_path}')
        return output_path

    def write_orientations_only(self, orientations, output_path, comment=''):
        """
        Write an orientations-only file for EMEBSD with anglefiletype='orientations'.
        This generates no strain — useful for pure orientation learning.
        """
        N = len(orientations)
        lines = []
        lines.append('eu')
        lines.append(str(N))
        if comment:
            lines.append(f'! {comment}')

        for i in range(N):
            e = orientations[i]
            lines.append(f'{e[0]:10.4f} {e[1]:10.4f} {e[2]:10.4f}')

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        print(f'[OK] Wrote {N} orientations to: {output_path}')
        return output_path

    def summary(self):
        """Print generation summary."""
        print(f'\n{"="*55}')
        print(f'  Angle/Strain Generator Summary')
        print(f'{"="*55}')
        print(f'  Patterns      : {self.n_patterns:,}')
        print(f'  Strain type   : {self.strain_type}')
        print(f'  Max magnitude : {self.strain_magnitude:.4f} ({self.strain_magnitude*100:.2f}%)')
        print(f'  Pattern center: xpc={self.xpc}, ypc={self.ypc}, L={self.L}')
        print(f'{"="*55}\n')


class SpatialFieldGenerator:
    """
    Generates spatially-correlated (kinematic) strain fields on a 2D scan grid
    for Stage 2 / Stage 3 training data.

    All fields are Level-1 kinematic: analytically defined displacement fields
    u(x,y) from which strain is computed as ε = sym(∇u).  They are
    geometrically compatible by construction and use strain magnitudes that are
    physically realistic for elastic loading of engineering metals (Fe/Cu/Ni).

    Flat index i maps to scan position (i // grid_cols, i % grid_cols) —
    row-major C order, the same convention used by train_grid.py / train_pair.py.

    Available field types
    ---------------------
    uniaxial_gradient
        Linear tension/compression along the horizontal scan axis with Poisson
        contraction in the transverse directions.
        Physical case: tensile specimen scanned along the loading axis.
        ε11 ∈ [5e-4, 3e-3], ε22 = ε33 = -ν · ε11.

    pure_bending
        Strain varies linearly through the scan height, zero at the neutral
        axis, ±ε_max at the top and bottom surfaces.
        Physical case: 3- or 4-point bending specimen, scan through cross-section.
        ε11 ∈ [-5e-3, +5e-3], ε22 = ε33 = -ν · ε11.

    gaussian_inclusion
        Localised strain concentration that decays as a Gaussian from a
        randomly placed centre.
        Physical case: hard carbide / precipitate in a steel matrix; Hertzian
        contact field under a nano-indenter; stress concentration at a void.
        ε_max ∈ [2e-3, 8e-3].

    shear_gradient
        Spatially varying shear (ε12 / ε13) with a linear or sinusoidal profile.
        Physical case: torsion bar cross-section; simple shear with spatial
        constraint; shear-band precursor.
        γ_max ∈ [5e-4, 4e-3].

    biaxial_gradient
        Equibiaxial strain (ε11 = ε22) with a radial profile centred near the
        scan origin.
        Physical case: thin-film thermal mismatch on cooling; biaxial tension
        of a plate with a stress concentrator.
        ε_max ∈ [5e-4, 4e-3].

    combined
        Weighted superposition of the above fields with randomly drawn weights.
        Produces the most varied training data and is recommended for general use.

    Orientation options
    -------------------
    constant_orientation=True   All patterns share one base orientation.
                                Suitable for isolating the strain signal.
    constant_orientation=False  Orientation drifts smoothly across the scan
                                (bilinear + small noise), mimicking a single
                                subgrain or slight low-angle tilt boundary.
                                Typical spread: 0.5–2° across a 100×100 scan.
    """

    # Poisson's ratio: Fe 0.29, Cu 0.34, Ni 0.31 — use Fe default.
    POISSON = 0.29

    FIELD_TYPES = [
        "uniaxial_gradient",
        "pure_bending",
        "gaussian_inclusion",
        "shear_gradient",
        "biaxial_gradient",
        "combined",
    ]

    def __init__(self, grid_rows: int, grid_cols: int, seed=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.N = grid_rows * grid_cols
        self.rng = np.random.default_rng(seed)

        # Normalised coordinate grids: C ∈ [0,1] (columns/x), R ∈ [0,1] (rows/y)
        cols = np.linspace(0.0, 1.0, grid_cols)
        rows = np.linspace(0.0, 1.0, grid_rows)
        self.C, self.R = np.meshgrid(cols, rows)  # both (grid_rows, grid_cols)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strain_to_F(self, eps_map: np.ndarray) -> np.ndarray:
        """
        eps_map: (grid_rows, grid_cols, 6) Voigt strain [ε11 ε22 ε33 ε23 ε13 ε12]
        returns: (N, 3, 3) deformation tensors F = I + ε, row-major order.
        Convention matches AngleStrainGenerator._strain_to_F and loader._F_to_voigt.
        """
        eps = eps_map.reshape(self.N, 6)
        F = np.zeros((self.N, 3, 3), dtype=np.float64)
        F[:, 0, 0] = 1.0 + eps[:, 0]   # F11
        F[:, 1, 1] = 1.0 + eps[:, 1]   # F22
        F[:, 2, 2] = 1.0 + eps[:, 2]   # F33
        F[:, 2, 1] = eps[:, 3]          # F32 ← ε23
        F[:, 1, 2] = eps[:, 3]          # F23
        F[:, 2, 0] = eps[:, 4]          # F31 ← ε13
        F[:, 0, 2] = eps[:, 4]          # F13
        F[:, 1, 0] = eps[:, 5]          # F21 ← ε12
        F[:, 0, 1] = eps[:, 5]          # F12
        return F

    def _add_noise(self, eps_map: np.ndarray, noise_frac: float) -> np.ndarray:
        """
        Add spatially-correlated Gaussian noise as a fraction of peak amplitude.
        Uses a 5×5 box-blur pass to give the noise some spatial coherence,
        mimicking real measurement / crystal imperfection scatter.
        """
        peak = np.abs(eps_map).max()
        if peak < 1e-12 or noise_frac <= 0:
            return eps_map
        sigma = peak * noise_frac
        raw_noise = self.rng.normal(0.0, sigma, eps_map.shape)
        # Light spatial smoothing: 5-tap MA along rows then cols (edge-pad so small
        # grids work; np.convolve mode='same' does not match length when n < kernel).
        k = np.ones(5) / 5.0

        def _ma5_1d(a: np.ndarray) -> np.ndarray:
            p = np.pad(a.astype(np.float64, copy=False), (2, 2), mode='edge')
            return np.convolve(p, k, mode='valid')

        for comp in range(6):
            for r in range(self.grid_rows):
                raw_noise[r, :, comp] = _ma5_1d(raw_noise[r, :, comp])
            for c in range(self.grid_cols):
                raw_noise[:, c, comp] = _ma5_1d(raw_noise[:, c, comp])
        return eps_map + raw_noise

    # ------------------------------------------------------------------
    # Individual field constructors
    # ------------------------------------------------------------------

    def _uniaxial_gradient(self, scale: float = 1.0) -> np.ndarray:
        """
        Linear uniaxial tension/compression along x (column direction) with
        Poisson contraction in y and z.

        u_x = α · x  →  ε11 = α,  ε22 = ε33 = -ν · α
        α ramps from 0 at the left edge to ε_max at the right edge.
        """
        eps_max = self.rng.uniform(5e-4, 3e-3) * scale
        sign = self.rng.choice([-1.0, 1.0])

        eps = np.zeros((self.grid_rows, self.grid_cols, 6))
        eps[:, :, 0] = sign * eps_max * self.C          # ε11
        eps[:, :, 1] = -self.POISSON * eps[:, :, 0]     # ε22
        eps[:, :, 2] = -self.POISSON * eps[:, :, 0]     # ε33
        return eps

    def _pure_bending(self, scale: float = 1.0) -> np.ndarray:
        """
        Pure bending of a beam with neutral axis at random height.
        ε11 varies linearly through the cross-section height (row direction).

        M / EI = κ  →  ε11(y) = κ · (y - y_neutral)
        Neutral axis placed at 40–60% of scan height (slight asymmetry allowed).
        """
        eps_max = self.rng.uniform(1e-3, 5e-3) * scale
        neutral = self.rng.uniform(0.40, 0.60)  # normalised height of neutral axis

        # Signed distance from neutral axis, scaled so surface strain = ±eps_max
        y_span = max(neutral, 1.0 - neutral)
        eps = np.zeros((self.grid_rows, self.grid_cols, 6))
        eps[:, :, 0] = eps_max * (self.R - neutral) / y_span   # ε11
        eps[:, :, 1] = -self.POISSON * eps[:, :, 0]             # ε22
        eps[:, :, 2] = -self.POISSON * eps[:, :, 0]             # ε33
        return eps

    def _gaussian_inclusion(self, scale: float = 1.0) -> np.ndarray:
        """
        Localised strain concentration decaying as a Gaussian from a centre.
        Includes both equibiaxial normal strains and a shear component.

        Models: hard carbide in steel, nano-indentation contact field, void tip.
        Inclusion placed in the interior (30–70% of scan) to ensure gradients
        are visible across the scan window.
        """
        eps_max = self.rng.uniform(2e-3, 8e-3) * scale
        cx = self.rng.uniform(0.30, 0.70)
        cy = self.rng.uniform(0.30, 0.70)
        sigma = self.rng.uniform(0.10, 0.25)   # width as fraction of scan size

        r2 = (self.C - cx) ** 2 + (self.R - cy) ** 2
        decay = np.exp(-r2 / (2.0 * sigma ** 2))

        biaxial_frac = self.rng.uniform(0.6, 1.0)
        shear_frac = self.rng.uniform(0.0, 0.35)

        eps = np.zeros((self.grid_rows, self.grid_cols, 6))
        eps[:, :, 0] = eps_max * biaxial_frac * decay
        eps[:, :, 1] = eps_max * biaxial_frac * decay
        eps[:, :, 2] = -2.0 * self.POISSON * eps[:, :, 0]
        eps[:, :, 5] = eps_max * shear_frac * decay    # ε12
        return eps

    def _shear_gradient(self, scale: float = 1.0) -> np.ndarray:
        """
        Spatially varying shear (ε12) with a linear or sinusoidal profile.
        A secondary out-of-plane shear (ε13 or ε23) at ~30% amplitude is included
        to reflect the fact that torsional / constrained-shear problems rarely
        activate a single component in isolation.

        Models: torsion bar cross-section scan; shear-band precursor zone;
        constrained simple shear specimen.
        """
        gamma_max = self.rng.uniform(5e-4, 4e-3) * scale
        mode = self.rng.choice(['linear', 'sinusoidal'])
        secondary_frac = self.rng.uniform(0.1, 0.35)

        eps = np.zeros((self.grid_rows, self.grid_cols, 6))
        if mode == 'linear':
            eps[:, :, 5] = gamma_max * (self.C - 0.5)              # ε12 linear in x
            eps[:, :, 4] = gamma_max * secondary_frac * (self.R - 0.5)  # ε13
        else:
            eps[:, :, 5] = gamma_max * np.sin(np.pi * self.C)      # ε12 sinusoidal
            eps[:, :, 3] = gamma_max * secondary_frac * np.sin(np.pi * self.R)  # ε23
        return eps

    def _biaxial_gradient(self, scale: float = 1.0) -> np.ndarray:
        """
        Equibiaxial strain (ε11 = ε22) with a radial profile from a central point.
        Either decays outward (high at centre → free surface) or rises outward
        (low at centre → inclusion constraint), both physically realistic.

        Models: thin-film cooling mismatch; Hertz contact pressure field;
        biaxial pressure-vessel geometry.
        """
        eps_max = self.rng.uniform(5e-4, 4e-3) * scale
        cx = self.rng.uniform(0.35, 0.65)
        cy = self.rng.uniform(0.35, 0.65)

        d = np.sqrt((self.C - cx) ** 2 + (self.R - cy) ** 2)
        # Normalise to [0, 1] using corner distance so profile fills the scan
        d_corner = np.sqrt(max(cx, 1 - cx) ** 2 + max(cy, 1 - cy) ** 2)
        d_norm = np.clip(d / d_corner, 0.0, 1.0)

        profile = (1.0 - d_norm) if self.rng.random() > 0.5 else d_norm

        eps = np.zeros((self.grid_rows, self.grid_cols, 6))
        eps[:, :, 0] = eps_max * profile
        eps[:, :, 1] = eps_max * profile
        eps[:, :, 2] = -2.0 * self.POISSON * eps[:, :, 0]
        return eps

    def _combined(self, scale: float = 1.0) -> np.ndarray:
        """
        Random weighted superposition of multiple field types.
        At least two fields are active; weights are drawn from U[0,1] and
        normalised.  Inclusion and shear amplitudes are halved in the mixture
        to prevent the combined field from exceeding physical strain limits.
        """
        candidates = [
            self._uniaxial_gradient(scale),
            self._pure_bending(scale),
            self._gaussian_inclusion(scale * 0.5),
            self._shear_gradient(scale * 0.5),
            self._biaxial_gradient(scale * 0.8),
        ]
        weights = self.rng.uniform(0.0, 1.0, len(candidates))
        # Force at least 2 non-zero contributors
        active = self.rng.choice(len(candidates), size=2, replace=False)
        mask = np.zeros(len(candidates))
        mask[active] = 1.0
        weights = weights * mask + weights * self.rng.uniform(0, 0.3, len(candidates))
        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum()
        return sum(w * f for w, f in zip(weights, candidates))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        field_type: str = "combined",
        scale: float = 1.0,
        constant_orientation: bool = True,
        base_euler: np.ndarray | None = None,
        orientation_spread_deg: float = 1.0,
        noise_frac: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a spatially correlated strain field + orientation array.

        Args:
            field_type:             One of FIELD_TYPES.
            scale:                  Multiplier on strain magnitudes (1.0 = physical defaults).
            constant_orientation:   All patterns share one orientation if True.
                                    If False, orientation drifts ~orientation_spread_deg
                                    across the scan (single subgrain / tilt boundary).
            base_euler:             (3,) Bunge Euler angles in degrees for the base orientation.
                                    Drawn from a single-crystal-like distribution if None.
            orientation_spread_deg: RMS misorientation spread in degrees across the scan.
                                    Typical: 0.5–2° for a subgrain. Only used when
                                    constant_orientation=False.
            noise_frac:             Gaussian noise on strain field as a fraction of peak
                                    amplitude (adds physical imperfection scatter).
                                    0.05 = 5%. Set to 0 for perfectly smooth fields.

        Returns:
            orientations:  (N, 3) Euler angles in degrees, row-major order.
            F_tensors:     (N, 3, 3) deformation tensors F = I + ε, row-major.
            eps_field:     (grid_rows, grid_cols, 6) Voigt strain field for inspection.
        """
        # ── Orientation field ──────────────────────────────────────────────
        if base_euler is None:
            # Reasonable single-crystal base: φ1 and φ2 fully random,
            # Φ (tilt) kept below 45° to avoid near-degenerate patterns.
            base_euler = np.array([
                self.rng.uniform(0.0, 360.0),
                self.rng.uniform(0.0, 45.0),
                self.rng.uniform(0.0, 360.0),
            ])

        if constant_orientation:
            orientations = np.tile(base_euler, (self.N, 1))
        else:
            # Bilinear drift across the scan + spatially correlated micro-noise.
            spread = orientation_spread_deg
            drift_x = self.rng.uniform(-spread, spread, size=3)
            drift_y = self.rng.uniform(-spread, spread, size=3)
            drift = (self.C[:, :, None] * drift_x
                     + self.R[:, :, None] * drift_y)          # (rows, cols, 3)
            micro_noise = self.rng.normal(0.0, spread * 0.08, drift.shape)
            euler_2d = base_euler + drift + micro_noise
            euler_2d[:, :, 0] %= 360.0
            euler_2d[:, :, 1] = np.clip(euler_2d[:, :, 1], 0.0, 180.0)
            euler_2d[:, :, 2] %= 360.0
            orientations = euler_2d.reshape(self.N, 3)

        # ── Strain field ───────────────────────────────────────────────────
        dispatch = {
            "uniaxial_gradient": self._uniaxial_gradient,
            "pure_bending":       self._pure_bending,
            "gaussian_inclusion": self._gaussian_inclusion,
            "shear_gradient":     self._shear_gradient,
            "biaxial_gradient":   self._biaxial_gradient,
            "combined":           self._combined,
        }
        if field_type not in dispatch:
            raise ValueError(
                f"Unknown field_type {field_type!r}. "
                f"Valid choices: {list(dispatch)}"
            )

        eps_field = dispatch[field_type](scale)          # (rows, cols, 6)
        if noise_frac > 0:
            eps_field = self._add_noise(eps_field, noise_frac)

        F_tensors = self._strain_to_F(eps_field)         # (N, 3, 3)
        return orientations, F_tensors, eps_field

    def print_field_stats(self, eps_field: np.ndarray) -> None:
        """Print per-component min / max / mean / std of the generated strain field."""
        VOIGT = ["ε11", "ε22", "ε33", "ε23", "ε13", "ε12"]
        flat = eps_field.reshape(-1, 6)
        print(f"\n{'Comp':>5}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
        print("  " + "-" * 52)
        for i, name in enumerate(VOIGT):
            v = flat[:, i]
            print(f"{name:>5}  {v.min():10.3e}  {v.max():10.3e}  "
                  f"{v.mean():10.3e}  {v.std():10.3e}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate orientations and strains for EBSD pattern synthesis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Shared args ────────────────────────────────────────────────────────────
    parser.add_argument('-n', '--n-patterns', type=int, default=10000,
                        help='Number of patterns (ignored when --spatial-field is set; '
                             'N is derived from grid-rows × grid-cols)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output file path (default: auto-generated from mode/type)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--xpc', type=float, default=0.0,
                        help='Pattern centre x in pixels (default: 0)')
    parser.add_argument('--ypc', type=float, default=0.0,
                        help='Pattern centre y in pixels (default: 0)')
    parser.add_argument('--L', type=float, default=15000.0,
                        help='Camera distance in microns (default: 15000)')
    parser.add_argument('--comment', type=str, default='',
                        help='Comment line written into the output file')

    # ── Stage 1 / random mode (existing behaviour) ────────────────────────────
    parser.add_argument('--strain-type', default='uniform',
                        choices=['uniform', 'uniaxial_x', 'uniaxial_y', 'uniaxial_z',
                                 'biaxial_xy', 'biaxial_yz', 'biaxial_xz',
                                 'multiaxial', 'shear_xy', 'shear_xz', 'shear_yz',
                                 'random', 'none'],
                        help='Strain type for random (non-spatial) mode')
    parser.add_argument('-s', '--strain-magnitude', type=float, default=0.0,
                        help='Maximum strain magnitude for random mode, e.g. 0.02 = 2%%')
    parser.add_argument('--uniform-strain', type=float, default=None,
                        help='Fixed hydrostatic strain value (overrides --strain-type)')
    parser.add_argument('--orient-only', action='store_true',
                        help='Output orientations only (no strain tensor)')

    # ── Stage 2 / 3 spatial mode ───────────────────────────────────────────────
    spatial = parser.add_argument_group(
        'Spatial field mode (for Stage 2 / Stage 3 training data)',
        'Pass --spatial-field to activate.  N is set by --grid-rows × --grid-cols.'
    )
    spatial.add_argument('--spatial-field', action='store_true',
                         help='Enable spatially-correlated strain field generation')
    spatial.add_argument('--grid-rows', type=int, default=100,
                         help='Number of scan rows (default: 100)')
    spatial.add_argument('--grid-cols', type=int, default=100,
                         help='Number of scan columns (default: 100)')
    spatial.add_argument('--field-type', default='combined',
                         choices=SpatialFieldGenerator.FIELD_TYPES,
                         help='Displacement field type (default: combined)')
    spatial.add_argument('--field-scale', type=float, default=1.0,
                         help='Scale factor on strain magnitudes (default: 1.0). '
                              'Reduce below 1.0 for smaller elastic strains.')
    spatial.add_argument('--constant-orientation', action='store_true',
                         help='All scan points share one fixed orientation '
                              '(default: orientation drifts slowly across the scan)')
    spatial.add_argument('--orientation-spread', type=float, default=1.0,
                         help='RMS misorientation spread in degrees across the scan '
                              'when orientation drift is active (default: 1.0°). '
                              'Typical for a single subgrain: 0.5–2°.')
    spatial.add_argument('--noise-frac', type=float, default=0.05,
                         help='Spatially-correlated noise added to strain field as a '
                              'fraction of peak amplitude (default: 0.05 = 5%%). '
                              'Set to 0 for perfectly smooth analytic fields.')

    args = parser.parse_args()

    # ── Spatial mode ────────────────────────────────────────────────────────────
    if args.spatial_field:
        gen_sp = SpatialFieldGenerator(
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            seed=args.seed,
        )
        N = gen_sp.N
        print(f'\n[spatial] grid = {args.grid_rows} × {args.grid_cols} = {N} patterns')
        print(f'[spatial] field type       = {args.field_type}')
        print(f'[spatial] field scale      = {args.field_scale}')
        print(f'[spatial] orientation mode = '
              f'{"constant" if args.constant_orientation else f"drifting ({args.orientation_spread}°)"}')
        print(f'[spatial] noise fraction   = {args.noise_frac}')

        orientations, F_tensors, eps_field = gen_sp.generate(
            field_type=args.field_type,
            scale=args.field_scale,
            constant_orientation=args.constant_orientation,
            orientation_spread_deg=args.orientation_spread,
            noise_frac=args.noise_frac,
        )
        gen_sp.print_field_stats(eps_field)

        # Reuse existing writer for xpc/ypc/L formatting
        writer = AngleStrainGenerator(n_patterns=N, seed=args.seed)
        writer.set_pattern_center(xpc=args.xpc, ypc=args.ypc, L=args.L)

        if args.output:
            output_path = args.output
        else:
            output_path = (f'data/raw/{args.grid_rows}x{args.grid_cols}'
                           f'_{args.field_type}_spatial.txt')

        comment = (args.comment or
                   f'Spatial field: {args.field_type}, '
                   f'grid={args.grid_rows}x{args.grid_cols}, '
                   f'scale={args.field_scale}, seed={args.seed}')
        writer.write_orpcdef(orientations, F_tensors, output_path, comment=comment)

        print(f'\nNext: set in EMEBSD.nml:')
        print(f"  anglefile = '{os.path.basename(output_path)}'")
        print(f"  anglefiletype = 'orpcdef'")
        print(f"  applyDeformation = 'y'")
        print(f"  ipf_wd = {args.grid_cols}  ! scan columns")
        print(f"  ipf_ht = {args.grid_rows}  ! scan rows")
        print(f'Then re-run: EMEBSD <path/to/EMEBSD.nml>')
        return

    # ── Random (Stage 1) mode — existing behaviour, unchanged ─────────────────
    gen = AngleStrainGenerator(n_patterns=args.n_patterns, seed=args.seed)

    strain_type = 'none' if args.orient_only else args.strain_type
    if args.uniform_strain is not None:
        strain_type = 'uniform'
        gen.uniform_strain_value = args.uniform_strain

    gen.set_pattern_center(xpc=args.xpc, ypc=args.ypc, L=args.L)
    orientations, F_tensors = gen.generate(
        strain_type=strain_type,
        strain_magnitude=args.strain_magnitude,
        uniform_strain_value=args.uniform_strain or 0.0,
    )
    gen.summary()

    if args.output:
        output_path = args.output
    else:
        base = f'{args.n_patterns:06d}'
        if args.orient_only:
            output_path = f'data/raw/{base}_orientations.txt'
        else:
            output_path = f'data/raw/{base}_strain_{args.strain_type}.txt'

    comment = args.comment or f'Generated by generate_angles.py ({args.n_patterns} patterns)'

    if args.orient_only:
        gen.write_orientations_only(orientations, output_path, comment=comment)
    else:
        gen.write_orpcdef(orientations, F_tensors, output_path, comment=comment)

    print(f'\nNext: set in EMEBSD.nml:')
    print(f"  anglefile = '{os.path.basename(output_path)}'")
    print(f"  anglefiletype = 'orpcdef'")
    print(f"  applyDeformation = 'y'")
    print(f'Then re-run: EMEBSD <path/to/EMEBSD.nml>')


if __name__ == '__main__':
    main()