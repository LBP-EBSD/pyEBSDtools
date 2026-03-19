#!/usr/bin/env python3
"""
EMsoft EBSD Data Loader
=======================
Utility for loading and inspecting EBSD synthetic pattern data from EMsoft HDF5 files.
Use this as a library in your ML pipeline or Jupyter notebooks.

Usage (as library):
    from load_data import EBSDDataLoader
    loader = EBSDDataLoader('/path/to/Fe_EBSD_patterns.h5')
    X = loader.get_patterns()       # (N, H, W)
    y = loader.get_euler()         # (N, 3) in degrees
    strain = loader.get_strain()    # (N, 3, 3) or None
    loader.to_numpy('/output/')     # dump to .npy files

Usage (as script):
    python load_data.py /path/to/Fe_EBSD_patterns.h5
"""

import os
import sys
import argparse

import h5py
import numpy as np


class EBSDDataLoader:
    """
    Loads EBSD pattern data from EMsoft HDF5 files.

    Attributes:
        file_path (str): Path to the HDF5 file.
        patterns (np.ndarray): Shape (N, H, W), float32. The pattern images.
        euler_angles (np.ndarray): Shape (N, 3), float64. Euler angles in degrees.
        deformation_tensors (np.ndarray): Shape (N, 3, 3) or None. F = I + ε tensors.
        pattern_centers (np.ndarray): Shape (N, 2) or None. Pattern center (xpc, ypc).
        crystal_name (str): Crystal structure name from the file.
        num_angles (int): Number of patterns.

    Example:
        >>> loader = EBSDDataLoader('Fe_EBSD_patterns.h5')
        >>> print(loader)  # summary
        >>> X = loader.patterns          # images for training
        >>> y = loader.euler_angles      # orientation labels
        >>> X_flat = loader.get_flattened()  # (N, H*W) for classic ML
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self._load()

    def _load(self):
        """Load all data from HDF5 file."""
        with h5py.File(self.file_path, 'r') as f:
            try:
                self.patterns = f['EMData/EBSD/EBSDPatterns'][:]
            except KeyError:
                raise ValueError(f"'EBSDPatterns' dataset not found in {self.file_path}. "
                                 f"Available groups: {list(f.keys())}")

            try:
                self.euler_angles = f['EMData/EBSD/EulerAngles'][:]
            except KeyError:
                self.euler_angles = None

            try:
                numangles = f['EMData/EBSD/numangles'][()]
                self.num_angles = int(numangles[0]) if hasattr(numangles, '__iter__') else int(numangles)
            except KeyError:
                self.num_angles = self.patterns.shape[0]

            try:
                self.crystal_name = self._read_str(f['EMData/EBSD/xtalname'])
            except KeyError:
                self.crystal_name = 'unknown'

            self.pattern_centers = None
            self.deformation_tensors = None

            try:
                pc = f['EMData/EBSD/PatternCenter'][:]
                self.pattern_centers = pc
            except KeyError:
                pass

            try:
                dt = f['EMData/EBSD/DeformationTensor'][:]
                self.deformation_tensors = dt
            except KeyError:
                pass

        if self.euler_angles is None:
            self.euler_angles = np.zeros((self.patterns.shape[0], 3))

        self.euler_angles = np.asarray(self.euler_angles, dtype=np.float64)

    @staticmethod
    def _read_str(ds):
        """Read a string dataset from HDF5."""
        if isinstance(ds[()], bytes):
            return ds[()].decode('utf-8').strip()
        return str(ds[()]).strip()

    def get_patterns(self):
        """Return patterns as (N, H, W) numpy array."""
        return self.patterns

    def get_euler(self):
        """Return Euler angles as (N, 3) numpy array in degrees."""
        return self.euler_angles

    def get_strain(self):
        """Return deformation tensors as (N, 3, 3) or None."""
        return self.deformation_tensors

    def get_flattened(self):
        """Return patterns flattened as (N, H*W) for classic ML."""
        return self.patterns.reshape(self.patterns.shape[0], -1)

    def get_normalized(self, method='minmax'):
        """
        Return normalized patterns.

        Args:
            method: 'minmax' (0-1), 'zscore' (zero mean, unit var),
                    or 'percentile' (1-99th percentile stretch).
        """
        p = self.patterns.copy()
        if method == 'minmax':
            pmin, pmax = p.min(), p.max()
            return (p - pmin) / (pmax - pmin)
        elif method == 'zscore':
            mean = p.mean()
            std = p.std()
            return (p - mean) / (std + 1e-8)
        elif method == 'percentile':
            lo, hi = np.percentile(p.ravel(), [1, 99])
            return np.clip((p - lo) / (hi - lo), 0, 1)
        raise ValueError(f'Unknown normalization method: {method}')

    def get_quaternions(self):
        """
        Convert Euler angles (Bunge convention) to quaternions.
        Returns (N, 4) array of quaternions [q0, q1, q2, q3].
        """
        from numpy.linalg import norm

        phi1, Phi, phi2 = np.deg2rad(self.euler_angles[:, 0]), \
                           np.deg2rad(self.euler_angles[:, 1]), \
                           np.deg2rad(self.euler_angles[:, 2])

        c1, s1 = np.cos(phi1 / 2), np.sin(phi1 / 2)
        c2, s2 = np.cos(Phi / 2), np.sin(Phi / 2)
        c3, s3 = np.cos(phi2 / 2), np.sin(phi2 / 2)

        q0 = c1 * c2 * c3 + s1 * s2 * s3
        q1 = s1 * c2 * c3 - c1 * s2 * s3
        q2 = c1 * s2 * c3 + s1 * c2 * s3
        q3 = c1 * c2 * s3 - s1 * s2 * c3

        q = np.stack([q0, q1, q2, q3], axis=1)
        return q / norm(q, axis=1, keepdims=True)

    def get_voigt_strain(self):
        """
        Extract Voigt-format elastic strain tensors from deformation tensors.
        Returns (N, 6) array [ε11, ε22, ε33, ε23, ε13, ε12].
        Returns None if deformation tensors are not available.
        """
        if self.deformation_tensors is None:
            return None

        eps = []
        for F in self.deformation_tensors:
            I = np.eye(3)
            E = 0.5 * (F.T @ F - I)
            voigt = [E[0, 0], E[1, 1], E[2, 2],
                     2 * E[1, 2], 2 * E[0, 2], 2 * E[0, 1]]
            eps.append(voigt)
        return np.array(eps)

    def to_numpy(self, output_dir):
        """
        Dump patterns and labels as .npy files for easy ML loading.

        Files saved:
            X_patterns.npy   — (N, H, W) float32
            y_euler.npy     — (N, 3) float64
            y_quaternion.npy — (N, 4) float64
            y_strain.npy    — (N, 6) float64 (Voigt format) or None
        """
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, '')

        np.save(base + 'X_patterns.npy', self.patterns.astype(np.float32))
        print(f'[OK] Saved: {base}X_patterns.npy  shape={self.patterns.shape}')

        np.save(base + 'y_euler.npy', self.euler_angles)
        print(f'[OK] Saved: {base}y_euler.npy    shape={self.euler_angles.shape}')

        q = self.get_quaternions()
        np.save(base + 'y_quaternion.npy', q)
        print(f'[OK] Saved: {base}y_quaternion.npy shape={q.shape}')

        strain = self.get_voigt_strain()
        if strain is not None:
            np.save(base + 'y_strain.npy', strain)
            print(f'[OK] Saved: {base}y_strain.npy   shape={strain.shape}')

    def to_tensorflow_dataset(self, batch_size=32, normalize='minmax'):
        """
        Return a tf.data.Dataset ready for training.
        Requires tensorflow installed.
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError('tensorflow not installed. Run: pip install tensorflow')

        X = self.get_normalized(normalize).astype(np.float32)
        y = self.get_quaternions().astype(np.float32)

        ds = tf.data.Dataset.from_tensor_slices((X, y))
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def summary(self):
        """Print a summary of the loaded data."""
        s = [
            f'EBSD Data Loader — {self.file_path}',
            f'  Patterns      : {self.patterns.shape}  dtype={self.patterns.dtype}',
            f'  Euler angles : {self.euler_angles.shape}  dtype={self.euler_angles.dtype}',
            f'  Pattern size : {self.pattern_shape[0]}x{self.pattern_shape[1]} pixels',
            f'  Crystal name : {self.crystal_name}',
            f'  Intensity    : [{self.patterns.min():.4f}, {self.patterns.max():.4f}]',
            f'  Strain data  : {"available" if self.deformation_tensors is not None else "not available"}',
        ]
        return '\n'.join(s)

    def __repr__(self):
        return self.summary()

    @property
    def pattern_shape(self):
        return (self.patterns.shape[1], self.patterns.shape[2])

    @property
    def num_patterns(self):
        return self.patterns.shape[0]

    @property
    def data_description(self):
        """Return a dict with all data shapes, useful for ML pipeline configs."""
        desc = {
            'file_path': self.file_path,
            'num_patterns': self.num_patterns,
            'pattern_height': self.pattern_shape[0],
            'pattern_width': self.pattern_shape[1],
            'total_pixels': self.pattern_shape[0] * self.pattern_shape[1],
            'euler_shape': self.euler_angles.shape,
            'has_strain': self.deformation_tensors is not None,
            'crystal_name': self.crystal_name,
        }
        return desc


def inspect_file(file_path):
    """Print the full HDF5 tree structure of an EBSD file."""
    print(f'\nFile: {file_path}')
    print(f'Size: {os.path.getsize(file_path) / 1e6:.1f} MB')
    print('-' * 60)

    def print_tree(name, obj):
        prefix = '  '
        if isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f'{prefix}{name:<45} {str(shape):<20} {str(dtype)}')
        elif isinstance(obj, h5py.Group):
            print(f'{prefix}[GROUP] {name}')

    with h5py.File(file_path, 'r') as f:
        print(f'{"Dataset":<45} {"Shape":<20} {"dtype"}')
        print('-' * 60)
        f.visititems(print_tree)


def main():
    parser = argparse.ArgumentParser(description='Load and inspect EMsoft EBSD data.')
    parser.add_argument('data_path', nargs='?', default=None)
    parser.add_argument('--inspect', '-i', action='store_true',
                        help='Show HDF5 file structure')
    parser.add_argument('--dump', '-d', metavar='DIR',
                        help='Dump patterns as .npy files to DIR')
    parser.add_argument('--info', action='store_true',
                        help='Show full HDF5 tree structure')
    args = parser.parse_args()

    if args.data_path is None:
        default = '/home/cosign/EMsoftData/Fe_FCC_exp/Fe_EBSD_patterns.h5'
        args.data_path = input(f'Enter path to EBSD HDF5 file [{default}]: ') or default

    if not os.path.exists(args.data_path):
        print(f'[ERROR] File not found: {args.data_path}')
        sys.exit(1)

    if args.info:
        inspect_file(args.data_path)
        return

    loader = EBSDDataLoader(args.data_path)
    print()
    print(loader.summary())

    if args.dump:
        loader.to_numpy(args.dump)

    print()


if __name__ == '__main__':
    main()
