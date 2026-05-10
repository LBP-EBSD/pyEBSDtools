"""
Crystal math utilities shared by the data-generation and training pipelines.

Functions here are pure numpy — no torch, no h5py dependencies.
Both datagen/convert.py and src/lbp_kikuchi/data/dataset.py (training side)
should import from here rather than reimplementing these conversions.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Euler ↔ Quaternion
# ─────────────────────────────────────────────────────────────────────────────

def euler_to_quaternion(euler_deg: np.ndarray) -> np.ndarray:
    """
    Convert Bunge (ZXZ) Euler angles to unit quaternions.

    Uses the scalar-first convention: q = [q0, q1, q2, q3]
    where q0 is the real (scalar) part.

    Euler angle ranges (Bunge convention):
        φ₁  (phi1) ∈ [0°, 360°]   — first rotation about Z
        Φ   (Phi)  ∈ [0°, 180°]   — rotation about X'
        φ₂  (phi2) ∈ [0°, 360°]   — second rotation about Z''

    Args:
        euler_deg: (N, 3) or (3,) array of Euler angles in degrees [phi1, Phi, phi2].

    Returns:
        (N, 4) or (4,) array of unit quaternions [q0, q1, q2, q3].
    """
    single = euler_deg.ndim == 1
    e = np.atleast_2d(euler_deg).astype(np.float64)

    phi1 = np.deg2rad(e[:, 0])
    Phi  = np.deg2rad(e[:, 1])
    phi2 = np.deg2rad(e[:, 2])

    c1, s1 = np.cos(phi1 / 2), np.sin(phi1 / 2)
    c2, s2 = np.cos(Phi  / 2), np.sin(Phi  / 2)
    c3, s3 = np.cos(phi2 / 2), np.sin(phi2 / 2)

    q0 = c1 * c2 * c3 + s1 * s2 * s3
    q1 = s1 * c2 * c3 - c1 * s2 * s3
    q2 = c1 * s2 * c3 + s1 * c2 * s3
    q3 = c1 * c2 * s3 - s1 * s2 * c3

    q = np.stack([q0, q1, q2, q3], axis=1)
    # normalise to guard against floating-point drift
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    return q[0] if single else q


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternions to Bunge (ZXZ) Euler angles in degrees.

    Args:
        q: (N, 4) or (4,) unit quaternions [q0, q1, q2, q3].

    Returns:
        (N, 3) or (3,) Euler angles in degrees [phi1, Phi, phi2].
    """
    single = q.ndim == 1
    q = np.atleast_2d(q).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)

    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Derived from the inverse of the ZXZ Bunge formula
    Phi  = 2.0 * np.arccos(np.clip(np.sqrt(q0**2 + q3**2), 0.0, 1.0))

    denom = np.sqrt(q1**2 + q2**2)
    safe  = denom > 1e-10

    phi1 = np.where(safe,
                    np.arctan2(q1 / np.where(safe, denom, 1.0),
                               -q2 / np.where(safe, denom, 1.0)),
                    np.arctan2(-q1 / np.where(~safe, denom + 1e-30, 1.0),
                               q2  / np.where(~safe, denom + 1e-30, 1.0)))

    denom2 = np.sqrt(q0**2 + q3**2)
    safe2  = denom2 > 1e-10
    phi2 = np.where(safe2,
                    np.arctan2(q1 / np.where(safe2, denom2, 1.0),
                               q2 / np.where(safe2, denom2, 1.0)),
                    np.zeros_like(q0))

    # Bring into [0, 2π)
    phi1 = phi1 % (2 * np.pi)
    phi2 = phi2 % (2 * np.pi)

    euler = np.stack([np.rad2deg(phi1), np.rad2deg(Phi), np.rad2deg(phi2)], axis=1)
    return euler[0] if single else euler


# ─────────────────────────────────────────────────────────────────────────────
# F tensor ↔ Voigt strain
# ─────────────────────────────────────────────────────────────────────────────

def ftensor_to_voigt(F: np.ndarray) -> np.ndarray:
    """
    Compute the Green-Lagrange strain tensor E = 0.5*(F^T F - I) and
    return it in Voigt notation.

    Voigt order: [ε₁₁, ε₂₂, ε₃₃, 2ε₂₃, 2ε₁₃, 2ε₁₂]
    (engineering shear components — factors of 2 on off-diagonals)

    Args:
        F: (N, 3, 3) or (3, 3) deformation gradient tensors.

    Returns:
        (N, 6) or (6,) Voigt strain.
    """
    single = F.ndim == 2
    F = np.atleast_3d(F) if single else F
    F = F.reshape(-1, 3, 3).astype(np.float64)

    I = np.eye(3, dtype=np.float64)
    # E_ij = 0.5 * (F^T F - I)  for each sample
    FtF = np.einsum('nki,nkj->nij', F, F)   # F^T @ F, vectorised
    E = 0.5 * (FtF - I)

    voigt = np.stack([
        E[:, 0, 0],        # ε₁₁
        E[:, 1, 1],        # ε₂₂
        E[:, 2, 2],        # ε₃₃
        2.0 * E[:, 1, 2],  # 2ε₂₃
        2.0 * E[:, 0, 2],  # 2ε₁₃
        2.0 * E[:, 0, 1],  # 2ε₁₂
    ], axis=1)

    return voigt[0] if single else voigt


def voigt_to_ftensor(eps: np.ndarray) -> np.ndarray:
    """
    Convert Voigt strain to a symmetric small-strain deformation tensor F = I + ε.

    For small strains (|ε| << 1) this is an accurate approximation.
    Voigt order: [ε₁₁, ε₂₂, ε₃₃, ε₂₃, ε₁₃, ε₁₂]
    (full engineering shear components, not doubled)

    Args:
        eps: (N, 6) or (6,) Voigt strain.

    Returns:
        (N, 3, 3) or (3, 3) deformation tensors.
    """
    single = eps.ndim == 1
    eps = np.atleast_2d(eps).astype(np.float64)
    N = len(eps)

    F = np.tile(np.eye(3, dtype=np.float64), (N, 1, 1))
    F[:, 0, 0] += eps[:, 0]    # F₁₁ = 1 + ε₁₁
    F[:, 1, 1] += eps[:, 1]    # F₂₂ = 1 + ε₂₂
    F[:, 2, 2] += eps[:, 2]    # F₃₃ = 1 + ε₃₃
    F[:, 1, 2] += eps[:, 3]    # F₂₃ = ε₂₃
    F[:, 2, 1] += eps[:, 3]    # symmetric
    F[:, 0, 2] += eps[:, 4]    # F₁₃ = ε₁₃
    F[:, 2, 0] += eps[:, 4]    # symmetric
    F[:, 0, 1] += eps[:, 5]    # F₁₂ = ε₁₂
    F[:, 1, 0] += eps[:, 5]    # symmetric

    return F[0] if single else F


# ─────────────────────────────────────────────────────────────────────────────
# Orientation sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_random_orientations(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n orientations uniformly distributed in SO(3) using Bunge Euler angles.

    Correct ranges for Bunge convention:
        φ₁  ∈ [0°, 360°]
        Φ   ∈ [0°, 180°]   ← NOT 360° — common sampling bug
        φ₂  ∈ [0°, 360°]

    Note: This gives a uniform distribution in SO(3) because the Haar measure
    for SO(3) in ZXZ Euler angles is sin(Φ) dΦ dφ₁ dφ₂. For a uniform prior
    on orientations this matters only if you are training orientation prediction
    with a non-symmetric crystal. For strain prediction it is less critical.

    Args:
        n:   Number of orientations.
        rng: numpy random Generator (for reproducibility).

    Returns:
        (n, 3) Euler angles in degrees.
    """
    phi1 = rng.uniform(0.0, 360.0, n)
    # Sample Φ with the correct Haar measure: Φ ~ arccos(U[−1,1]) → Φ ∈ [0°,180°]
    Phi  = np.rad2deg(np.arccos(rng.uniform(-1.0, 1.0, n)))
    phi2 = rng.uniform(0.0, 360.0, n)
    return np.stack([phi1, Phi, phi2], axis=1)
