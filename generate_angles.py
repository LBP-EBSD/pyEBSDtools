#!/usr/bin/env python3
"""
EMsoft EBSD Angle & Strain Generator
===================================
Generates the input file for EMEBSD pattern generation with random orientations
and/or elastic strains. Produces the .txt file that EMEBSD reads with
`anglefiletype = 'orpcdef'`.

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
        euler1 euler2 euler3 xpc ypc F11 F21 F31 F12 F22 F32 F13 F23 F33
        (F = deformation tensor = I + strain, column-major order)
"""

import argparse
import os
import sys
import math

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
            Line 3+: one line per pattern:
                euler1 euler2 euler3 xpc ypc F11 F21 F31 F12 F22 F32 F13 F23 F33
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
                    f'{self.xpc:6.2f} {self.ypc:6.2f} '
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
        print(f'  Pattern center: xpc={self.xpc}, ypc={self.ypc}')
        print(f'{"="*55}\n')


def main():
    parser = argparse.ArgumentParser(
        description='Generate random orientations and strains for EBSD pattern synthesis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('-n', '--n-patterns', type=int, default=10000,
                        help='Number of patterns to generate (default: 10000)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output file path (default: auto from strain type)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--strain-type', default='uniform',
                        choices=['uniform', 'uniaxial_x', 'uniaxial_y', 'uniaxial_z',
                                 'biaxial_xy', 'biaxial_yz', 'biaxial_xz',
                                 'multiaxial', 'shear_xy', 'shear_xz', 'shear_yz',
                                 'random', 'none'],
                        help='Type of elastic strain to apply (default: uniform/none)')
    parser.add_argument('-s', '--strain-magnitude', type=float, default=0.0,
                        help='Maximum strain magnitude, e.g., 0.02 = 2%% strain (default: 0.0)')
    parser.add_argument('--uniform-strain', type=float, default=None,
                        help='Fixed uniform (hydrostatic) strain value, e.g., 0.005')
    parser.add_argument('--xpc', type=float, default=0.0,
                        help='Pattern center x coordinate in pixels (default: 0)')
    parser.add_argument('--ypc', type=float, default=0.0,
                        help='Pattern center y coordinate in pixels (default: 0)')
    parser.add_argument('--L', type=float, default=15000.0,
                        help='Camera distance in microns (default: 15000)')
    parser.add_argument('--orient-only', action='store_true',
                        help='Output orientations only (no strain), for anglefiletype=orientations')
    parser.add_argument('--comment', type=str, default='',
                        help='Comment line added to output file')
    args = parser.parse_args()

    gen = AngleStrainGenerator(n_patterns=args.n_patterns, seed=args.seed)

    strain_type = 'none' if args.orient_only else args.strain_type
    if args.uniform_strain is not None:
        strain_type = 'uniform'
        gen.uniform_strain_value = args.uniform_strain

    gen.set_pattern_center(xpc=args.xpc, ypc=args.ypc, L=args.L)
    orientations, F_tensors = gen.generate(
        strain_type=strain_type,
        strain_magnitude=args.strain_magnitude,
        uniform_strain_value=args.uniform_strain or 0.0
    )
    gen.summary()

    if args.output:
        output_path = args.output
    else:
        base = f'{args.n_patterns:06d}'
        if args.orient_only:
            output_path = f'/home/cosign/EMsoftData/Fe_FCC_exp/{base}_orientations.txt'
        else:
            output_path = f'/home/cosign/EMsoftData/Fe_FCC_exp/{base}_strain_{args.strain_type}.txt'

    comment = args.comment or f'Generated by generate_angles.py ({args.n_patterns} patterns)'

    if args.orient_only:
        gen.write_orientations_only(orientations, output_path, comment=comment)
    else:
        gen.write_orpcdef(orientations, F_tensors, output_path, comment=comment)

    print(f'\nNext: use this file as "anglefile" in your EMEBSD.nml')
    print(f'  anglefile = \'{os.path.basename(output_path)}\'')
    print(f'  anglefiletype = \'orpcdef\'')
    print(f'  applyDeformation = \'y\'')
    print(f'\nTo regenerate patterns, re-run:')
    print(f'  EMEBSD /home/EMuser/EMPlay/Fe_FCC_exp/EMEBSD.nml')


if __name__ == '__main__':
    main()
