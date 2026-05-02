#!/usr/bin/env python3
"""
CLI entry point for angle / strain file generation.

The implementation lives in ``datagen/angle_generation.py`` (shared with
``make generate`` / ``datagen.sampler``). Run from the ``EBSDtools`` directory::

    python generate_angles.py --help
    python generate_angles.py -n 5000 -s 0.01 --strain-type multiaxial
"""

from __future__ import annotations

import sys
from pathlib import Path

_EBSDTOOLS = Path(__file__).resolve().parent
if str(_EBSDTOOLS) not in sys.path:
    sys.path.insert(0, str(_EBSDTOOLS))

if __name__ == "__main__":
    from datagen.angle_generation import main

    main()
