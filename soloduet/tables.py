"""Precomputed Jacobi-sequence and Roche-binary solution grids.

Both grids ship as ``.npz`` files inside ``soloduet/data/`` so importing and
fitting are instant and the numerics are pinned for regression testing.  They
are regenerated with ``scripts/build_tables.py``; if a file is missing the
grid is silently rebuilt in memory (never written at import time, so
read-only installs are safe).

Grid contents
-------------
``jacobi_sequence.npz`` : arrays ``b_over_a, c_over_a, omega2`` (200 nodes
    over b/a in [0.43, 1.00]).
``roche_binaries.npz``  : flat arrays over all matched solutions —
    ``q, b1, c1, a2, b2, c2, omega2, separation, mismatch`` (secondary axes
    physically rescaled to primary units; q in [0.05, 1.00] step 0.01).
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

JACOBI_NPZ = os.path.join(_DATA_DIR, "jacobi_sequence.npz")
ROCHE_NPZ = os.path.join(_DATA_DIR, "roche_binaries.npz")

#: default Roche mass-ratio grid (Lacerda & Jewitt 2007 use 0.01 steps)
Q_GRID = np.round(np.arange(0.05, 1.0 + 1e-9, 0.01), 2)


def build_jacobi(n: int = 200) -> dict:
    """Compute the Jacobi-sequence grid (see :func:`soloduet.jacobi.jacobi_sequence`)."""
    from .jacobi import jacobi_sequence

    return jacobi_sequence(n)


def build_roche(q_grid=None, verbose: bool = False) -> dict:
    """Compute the matched Roche-binary grid over ``q_grid``."""
    from .roche import match_binary

    if q_grid is None:
        q_grid = Q_GRID
    rows = {k: [] for k in
            ("q", "b1", "c1", "a2", "b2", "c2", "omega2", "separation", "mismatch")}
    for q in q_grid:
        sols = match_binary(float(q))
        if verbose:
            print(f"  q = {q:.2f}: {len(sols)} solutions")
        for s in sols:
            rows["q"].append(s.q)
            rows["b1"].append(s.primary_axes[1])
            rows["c1"].append(s.primary_axes[2])
            rows["a2"].append(s.secondary_axes[0])
            rows["b2"].append(s.secondary_axes[1])
            rows["c2"].append(s.secondary_axes[2])
            rows["omega2"].append(s.omega2)
            rows["separation"].append(s.separation)
            rows["mismatch"].append(s.mismatch)
    return {k: np.array(v, dtype=float) for k, v in rows.items()}


def build_all(outdir: str = _DATA_DIR, verbose: bool = True) -> None:
    """Build both grids and write the ``.npz`` files (used by scripts/build_tables.py)."""
    os.makedirs(outdir, exist_ok=True)
    if verbose:
        print("Building Jacobi sequence ...")
    np.savez_compressed(os.path.join(outdir, "jacobi_sequence.npz"), **build_jacobi())
    if verbose:
        print("Building Roche binary grid (a few minutes) ...")
    np.savez_compressed(os.path.join(outdir, "roche_binaries.npz"),
                        **build_roche(verbose=verbose))
    if verbose:
        print(f"Tables written to {outdir}")


@lru_cache(maxsize=1)
def load_jacobi() -> dict:
    """Load (or rebuild) the Jacobi-sequence grid as a dict of arrays."""
    if os.path.exists(JACOBI_NPZ):
        with np.load(JACOBI_NPZ) as z:
            return {k: z[k].copy() for k in z.files}
    return build_jacobi()


@lru_cache(maxsize=1)
def load_roche() -> dict:
    """Load (or rebuild) the Roche-binary grid as a dict of arrays."""
    if os.path.exists(ROCHE_NPZ):
        with np.load(ROCHE_NPZ) as z:
            return {k: z[k].copy() for k in z.files}
    return build_roche()


__all__ = [
    "load_jacobi",
    "load_roche",
    "build_jacobi",
    "build_roche",
    "build_all",
    "JACOBI_NPZ",
    "ROCHE_NPZ",
    "Q_GRID",
]
