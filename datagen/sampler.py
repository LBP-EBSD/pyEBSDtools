"""
Orientation and strain sampler — Stage 1 of the data generation pipeline.

Generates random Euler angles and deformation gradient tensors (F), writes:
  - <experiment_name>_angles.txt   — EMEBSD input file (orpcdef format)
  - <experiment_name>_Ftensors.npy — (N, 3, 3) F tensors saved as labels
  - <experiment_name>_euler.npy    — (N, 3) Euler angles saved as labels

Saving F tensors and Euler angles separately is essential because EMsoft does
NOT reliably write deformation tensors back into its output HDF5 file.

Output .txt format (EMEBSD anglefiletype='orpcdef'):
    Line 1: 'eu'
    Line 2: N
    Line 3+: euler1 euler2 euler3 xpc ypc F11 F21 F31 F12 F22 F32 F13 F23 F33
              (F written in column-major order as expected by EMEBSD)
"""

import os
import numpy as np
from helpers.crystal import sample_random_orientations, ftensor_to_voigt


# ─── Supported strain types ───────────────────────────────────────────────────
STRAIN_TYPES = [
    "uniform",
    "uniaxial_x", "uniaxial_y", "uniaxial_z",
    "biaxial_xy", "biaxial_yz", "biaxial_xz",
    "multiaxial",
    "shear_xy", "shear_xz", "shear_yz",
    "random",
]


class Sampler:
    """
    Samples random EBSD scan parameters: orientations + elastic strains.

    Args:
        n_patterns:       Number of patterns to generate.
        strain_type:      One of STRAIN_TYPES.
        strain_magnitude: Max |ε| for random strain generation (e.g. 0.02 = 2%).
        seed:             Integer seed for reproducibility.
        xpc:              Pattern center x offset in pixels (passed through to file).
        ypc:              Pattern center y offset in pixels.
    """

    def __init__(
        self,
        n_patterns: int = 10_000,
        strain_type: str = "multiaxial",
        strain_magnitude: float = 0.02,
        seed: int | None = None,
        xpc: float = 0.0,
        ypc: float = 0.0,
    ):
        if strain_type not in STRAIN_TYPES:
            raise ValueError(f"Unknown strain_type {strain_type!r}. Choose from: {STRAIN_TYPES}")

        self.n_patterns = n_patterns
        self.strain_type = strain_type
        self.strain_magnitude = strain_magnitude
        self.rng = np.random.default_rng(seed)
        self.xpc = xpc
        self.ypc = ypc

    # ─── Public ───────────────────────────────────────────────────────────────

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample orientations and F tensors.

        Returns:
            euler:    (N, 3)    Euler angles in degrees [phi1, Phi, phi2].
            F_tensors:(N, 3, 3) Deformation gradient tensors.
        """
        euler = sample_random_orientations(self.n_patterns, self.rng)
        F_tensors = np.stack([
            self._sample_F() for _ in range(self.n_patterns)
        ])
        return euler, F_tensors

    def save(self, out_dir: str, experiment_name: str) -> dict[str, str]:
        """
        Generate, then write all output files to `out_dir`.

        Returns:
            dict of {role: absolute_path} for every file written.
        """
        os.makedirs(out_dir, exist_ok=True)
        euler, F_tensors = self.generate()

        angles_path  = os.path.join(out_dir, f"{experiment_name}_angles.txt")
        ftensor_path = os.path.join(out_dir, f"{experiment_name}_Ftensors.npy")
        euler_path   = os.path.join(out_dir, f"{experiment_name}_euler.npy")

        self._write_orpcdef(euler, F_tensors, angles_path)
        np.save(ftensor_path, F_tensors.astype(np.float64))
        np.save(euler_path,   euler.astype(np.float64))

        print(f"[sampler] Wrote {self.n_patterns:,} samples:")
        print(f"  angles  → {angles_path}")
        print(f"  F tensors → {ftensor_path}  shape={F_tensors.shape}")
        print(f"  euler     → {euler_path}    shape={euler.shape}")

        return {
            "angles_txt":  angles_path,
            "ftensors_npy": ftensor_path,
            "euler_npy":    euler_path,
        }

    # ─── Strain generators ────────────────────────────────────────────────────

    def _sample_F(self) -> np.ndarray:
        """Sample one (3, 3) deformation tensor for the configured strain type."""
        t = self.strain_type
        mag = self.strain_magnitude

        if t == "uniform":
            return self._F_from_voigt(np.zeros(6))

        if t == "uniaxial_x":
            eps = np.zeros(6)
            eps[0] = self.rng.uniform(-mag, mag)
            return self._F_from_voigt(eps)

        if t == "uniaxial_y":
            eps = np.zeros(6)
            eps[1] = self.rng.uniform(-mag, mag)
            return self._F_from_voigt(eps)

        if t == "uniaxial_z":
            eps = np.zeros(6)
            eps[2] = self.rng.uniform(-mag, mag)
            return self._F_from_voigt(eps)

        if t == "biaxial_xy":
            val = self.rng.uniform(-mag, mag)
            eps = np.zeros(6); eps[0] = val; eps[1] = val
            return self._F_from_voigt(eps)

        if t == "biaxial_yz":
            val = self.rng.uniform(-mag, mag)
            eps = np.zeros(6); eps[1] = val; eps[2] = val
            return self._F_from_voigt(eps)

        if t == "biaxial_xz":
            val = self.rng.uniform(-mag, mag)
            eps = np.zeros(6); eps[0] = val; eps[2] = val
            return self._F_from_voigt(eps)

        if t == "shear_xy":
            eps = np.zeros(6)
            eps[5] = self.rng.uniform(-mag, mag)
            return self._F_from_voigt(eps)

        if t == "shear_xz":
            eps = np.zeros(6)
            eps[4] = self.rng.uniform(-mag, mag)
            return self._F_from_voigt(eps)

        if t == "shear_yz":
            eps = np.zeros(6)
            eps[3] = self.rng.uniform(-mag, mag)
            return self._F_from_voigt(eps)

        if t == "multiaxial":
            eps = self.rng.uniform(-mag, mag, 6)
            return self._F_from_voigt(eps)

        if t == "random":
            # Weighted mix: more multiaxial, some uniaxial/shear for variety
            choice = self.rng.choice(
                ["multiaxial", "uniaxial_x", "uniaxial_y", "uniaxial_z",
                 "biaxial_xy", "shear_xy", "shear_xz"],
                p=[0.40, 0.15, 0.10, 0.10, 0.10, 0.075, 0.075],
            )
            old_type = self.strain_type
            self.strain_type = choice
            F = self._sample_F()
            self.strain_type = old_type
            return F

        return self._F_from_voigt(np.zeros(6))

    @staticmethod
    def _F_from_voigt(eps: np.ndarray) -> np.ndarray:
        """Build F = I + ε from a Voigt vector (full engineering shear components)."""
        F = np.eye(3, dtype=np.float64)
        F[0, 0] += eps[0]
        F[1, 1] += eps[1]
        F[2, 2] += eps[2]
        F[1, 2] += eps[3]; F[2, 1] += eps[3]
        F[0, 2] += eps[4]; F[2, 0] += eps[4]
        F[0, 1] += eps[5]; F[1, 0] += eps[5]
        return F

    # ─── File writer ──────────────────────────────────────────────────────────

    def _write_orpcdef(
        self,
        euler: np.ndarray,
        F_tensors: np.ndarray,
        output_path: str,
    ) -> None:
        """
        Write the EMEBSD orientation+deformation input file (anglefiletype='orpcdef').

        Format per data line (column-major F):
            euler1 euler2 euler3 xpc ypc F11 F21 F31 F12 F22 F32 F13 F23 F33
        """
        lines = ["eu", str(self.n_patterns)]
        lines.append(
            f"! Generated by datagen/sampler.py — "
            f"n={self.n_patterns} strain={self.strain_type} mag={self.strain_magnitude}"
        )

        for i in range(self.n_patterns):
            e = euler[i]
            F = F_tensors[i]
            line = (
                f"{e[0]:10.4f} {e[1]:10.4f} {e[2]:10.4f} "
                f"{self.xpc:6.2f} {self.ypc:6.2f} "
                # column-major: col0, col1, col2
                f"{F[0,0]:14.8f} {F[1,0]:14.8f} {F[2,0]:14.8f} "
                f"{F[0,1]:14.8f} {F[1,1]:14.8f} {F[2,1]:14.8f} "
                f"{F[0,2]:14.8f} {F[1,2]:14.8f} {F[2,2]:14.8f}"
            )
            lines.append(line)

        with open(output_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")


# ─── CLI convenience ──────────────────────────────────────────────────────────

def run_from_config(cfg: dict) -> dict[str, str]:
    """
    Run the sampler from a parsed config dict (as loaded from config.yaml).

    Returns paths dict from Sampler.save().
    """
    gen_cfg   = cfg["generation"]
    paths_cfg = cfg["paths"]

    data_dir = os.path.expanduser(paths_cfg["data_dir"])
    exp_dir  = os.path.join(data_dir, paths_cfg["experiment_name"])

    sampler = Sampler(
        n_patterns       = gen_cfg["n_patterns"],
        strain_type      = gen_cfg["strain_type"],
        strain_magnitude = gen_cfg["strain_magnitude"],
        seed             = gen_cfg.get("seed"),
        xpc              = gen_cfg.get("xpc", 0.0),
        ypc              = gen_cfg.get("ypc", 0.0),
    )
    return sampler.save(exp_dir, paths_cfg["experiment_name"])
