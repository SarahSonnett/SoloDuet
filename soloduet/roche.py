"""Roche binary equilibrium figures.

Close and contact binary components are modelled with the Roche binary
approximation of Lacerda & Jewitt (2007), following Chandrasekhar (1963) and
Leone et al. (1984): each component is a *Roche ellipsoid* — the equilibrium
shape of a homogeneous, tidally locked satellite whose deformation is caused
by the spherically symmetric gravity of its companion.  Each component's
shape is computed separately, using mass ratios ``q`` and ``1/q`` (``q`` is
always the companion-to-self mass ratio in the component equations).

With semi-axes ``a = 1 >= b >= c`` and index symbols ``A_i``
(:mod:`soloduet.chandrasekhar`), a component satisfies (Lacerda & Jewitt
2007, eq. 4; Chandrasekhar 1963)

    [(3 + 1/q) a^2 + c^2] / [(1/q) b^2 + c^2]
        = (a^2 A_1 - c^2 A_3) / (b^2 A_2 - c^2 A_3)

    [q / (1+q)] omega^2/(pi G rho)
        = 2 (a^2 A_1 - c^2 A_3) / [(3 + 1/q) a^2 + c^2]

(the second equation is written for the abc-normalized index symbols used
throughout SoloDuet, ``A_1 + A_2 + A_3 = 2``; Lacerda & Jewitt print it with
an explicit ``abc`` factor because their A_i omit that normalization — the
shape equation is a ratio and is identical in both conventions)

A valid *binary* is a pair of component solutions — primary computed with
``q`` (= m2/m1 <= 1), secondary with ``1/q`` — that share the same
``omega^2/(pi G rho)``.  Both components are assumed tidally locked with
equal densities on a circular orbit; the light-curve period equals the
orbital period.  The orbital separation follows from Kepler's third law with
point masses, quoted as ``d / (a_1 + a_2)`` so that ~1 indicates contact.

Regression anchor: Lacerda & Jewitt (2007), Table 3 — for ``q = 0.25`` and
primary ``(B/A, C/A) = (0.91674, 0.83)`` the secondary is
``(b/a, c/a) = (0.51426, 0.48)`` with ``omega^2/(pi G rho) = 0.10626`` and
``d/(A + a) = 1.19222``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import brentq

from scipy.special import elliprd

from .chandrasekhar import index_symbols


def _index_symbols_vec(b: np.ndarray, c: float):
    """Vectorized index symbols for semi-axes (1, b_array, c)."""
    b2 = b * b
    c2 = c * c
    abc = b * c
    A1 = (2.0 / 3.0) * abc * elliprd(b2, c2, 1.0)
    A2 = (2.0 / 3.0) * abc * elliprd(1.0, c2, b2)
    A3 = (2.0 / 3.0) * abc * elliprd(1.0, b2, c2)
    return A1, A2, A3


@dataclass
class RocheComponent:
    """Equilibrium shape of one tidally locked component (semi-axis a = 1).

    ``q`` is the *companion-to-self* mass ratio used in the shape equations.
    """

    q: float
    b: float
    c: float
    omega2: float  # omega^2 / (pi G rho)

    @property
    def axes(self) -> np.ndarray:
        return np.array([1.0, self.b, self.c])


@dataclass
class BinarySolution:
    """A matched Roche binary (equal densities, tidally locked, circular).

    All lengths are in units of the primary's long semi-axis.  ``q`` is the
    secondary-to-primary mass ratio (<= 1).  ``mismatch`` is the relative
    difference between the two components' normalized spins: zero when the
    secondary curve was interpolated exactly, positive (but below the
    matching tolerance) for pairs at the secondary's Roche limit, where the
    approximation of Lacerda & Jewitt (2007) strains.
    """

    q: float
    primary_axes: np.ndarray    # (1, b1, c1)
    secondary_axes: np.ndarray  # (s, s*b2, s*c2), physically rescaled
    omega2: float               # omega^2 / (pi G rho)
    separation: float           # d / (a_primary + a_secondary); ~1 => contact
    mismatch: float = 0.0

    @property
    def scale(self) -> float:
        """Secondary long semi-axis in primary units."""
        return float(self.secondary_axes[0])

    @property
    def center_distance(self) -> float:
        """Center-to-center distance d in primary long-semi-axis units."""
        return float(self.separation * (1.0 + self.scale))


def _component_residual(b: float, q: float, c: float) -> float:
    """Residual of the Roche shape equation at semi-axes (1, b, c)."""
    A1, A2, A3 = index_symbols(1.0, b, c)
    lhs_den = (1.0 / q) * b * b + c * c
    rhs_den = b * b * A2 - c * c * A3
    lhs = ((3.0 + 1.0 / q) + c * c) / lhs_den
    rhs = (A1 - c * c * A3) / rhs_den
    return lhs - rhs


def _component_omega2(q: float, b: float, c: float) -> float:
    """Normalized spin from the second Roche equation (a = 1)."""
    A1, _, A3 = index_symbols(1.0, b, c)
    return ((1.0 + q) / q) * 2.0 * (A1 - c * c * A3) / ((3.0 + 1.0 / q) + c * c)


def roche_component(q: float, c: float, n_scan: int = 200) -> List[RocheComponent]:
    """Solve the Roche shape equation for ``b`` at fixed ``q`` and ``c``.

    Scans ``b`` in ``(c, 1)`` for sign changes and refines each root with
    Brent's method.  Returns all physical roots (``c <= b <= 1`` and
    ``omega2 > 0``), ordered by increasing ``b``; an empty list means no
    equilibrium exists (past the Roche limit for that ``q``).
    """
    if not (0.0 < c < 1.0):
        return []
    b_grid = np.linspace(c * (1.0 + 1e-9), 1.0, int(n_scan))
    A1, A2, A3 = _index_symbols_vec(b_grid, c)
    b2 = b_grid * b_grid
    c2 = c * c
    with np.errstate(invalid="ignore", divide="ignore"):
        lhs = ((3.0 + 1.0 / q) + c2) / ((1.0 / q) * b2 + c2)
        rhs = (A1 - c2 * A3) / (b2 * A2 - c2 * A3)
        vals = lhs - rhs
    roots: List[RocheComponent] = []
    for i in range(len(b_grid) - 1):
        v0, v1 = vals[i], vals[i + 1]
        if not (np.isfinite(v0) and np.isfinite(v1)) or v0 * v1 > 0.0:
            continue
        try:
            b_root = brentq(_component_residual, b_grid[i], b_grid[i + 1],
                            args=(q, c), xtol=1e-12)
        except ValueError:
            continue
        omega2 = _component_omega2(q, b_root, c)
        if omega2 > 0.0:
            roots.append(RocheComponent(q=q, b=float(b_root), c=float(c),
                                        omega2=float(omega2)))
    return roots


def component_sequence(q: float, c_grid: np.ndarray) -> List[RocheComponent]:
    """Trace the physical (least-elongated) solution branch over ``c_grid``.

    For each ``c`` (scanned from large to small), keeps the root whose ``b``
    continues the branch from the previous node; the branch terminates at the
    Roche limit, where no root remains.
    """
    comps: List[RocheComponent] = []
    prev_b: Optional[float] = None
    for c in np.sort(np.asarray(c_grid, dtype=float))[::-1]:
        roots = roche_component(q, float(c))
        if not roots:
            break
        if prev_b is None:
            pick = roots[-1]  # near-spherical end: largest-b root
        else:
            pick = min(roots, key=lambda r: abs(r.b - prev_b))
        comps.append(pick)
        prev_b = pick.b
    return comps[::-1]  # ascending c


def _stable_side(comps: List[RocheComponent], q: float,
                 refine: bool = True) -> List[RocheComponent]:
    """Truncate a component branch to its stable side, ``c >= c(spin peak)``.

    Along a branch traced in ``c``, the normalized spin rises to a maximum at
    the component's Roche limit and declines on the over-elongated
    continuation; only the near-spherical side of the peak is the physical
    satellite sequence.  When ``refine`` is set, the peak node is sharpened
    by a short sub-grid scan so the maximum matchable spin is accurate.
    """
    if len(comps) < 3:
        return comps
    w = np.array([s.omega2 for s in comps])  # comps are ascending in c
    i_pk = int(np.argmax(w))
    kept = comps[i_pk:]
    if refine and 0 < i_pk < len(comps) - 1:
        c_lo, c_hi = comps[i_pk - 1].c, comps[i_pk + 1].c
        best = kept[0]
        for c in np.linspace(c_lo, c_hi, 21):
            roots = roche_component(q, float(c))
            for r in roots:
                if abs(r.b - best.b) < 0.05 and r.omega2 > best.omega2:
                    best = r
        if best is not kept[0]:
            kept = [best] + [s for s in kept if s.c > best.c]
    return kept


def match_binary(q: float, c_step: float = 0.01, c_max: float = 0.99,
                 mismatch_tol: float = 0.02) -> List[BinarySolution]:
    """Matched Roche binary solutions for secondary/primary mass ratio ``q``.

    The primary branch is computed with companion ratio ``q``, the secondary
    branch with ``1/q``; both are truncated to their stable (near-spherical)
    sides, and the secondary's normalized-spin curve is interpolated to the
    primary's ``omega2`` at every primary node (Lacerda & Jewitt 2007).
    Primary nodes whose spin exceeds the secondary's Roche-limit maximum by
    less than ``mismatch_tol`` (relative) are paired with the limit node at
    the mean spin — this reproduces the tolerance implicit in Lacerda &
    Jewitt's 0.01-step grid matching (their Table 3) and is flagged via
    ``BinarySolution.mismatch``.  The secondary is rescaled to the
    equal-density volume (``V2 = q V1``) before the Kepler separation is
    computed.
    """
    q = float(q)
    if not (0.0 < q <= 1.0):
        raise ValueError("q must be in (0, 1] (secondary/primary mass ratio)")

    c_grid = np.arange(c_step, c_max + 0.5 * c_step, c_step)
    prim = _stable_side(component_sequence(q, c_grid), q)
    if not prim:
        return []

    if abs(q - 1.0) < 1e-12:
        sec = prim  # identical components
    else:
        sec = _stable_side(component_sequence(1.0 / q, c_grid), 1.0 / q)
    if not sec:
        return []

    # on the stable side omega2 decreases monotonically with c
    sec_c = np.array([s.c for s in sec])
    sec_b = np.array([s.b for s in sec])
    sec_w = np.array([s.omega2 for s in sec])
    order = np.argsort(sec_w)
    sec_w_sorted = sec_w[order]
    sec_c_sorted = sec_c[order]
    sec_b_sorted = sec_b[order]
    w_max = float(sec_w_sorted[-1])

    solutions: List[BinarySolution] = []
    for p in prim:
        mismatch = 0.0
        omega2 = p.omega2
        if p.omega2 < sec_w_sorted[0]:
            continue
        if p.omega2 <= w_max:
            c2 = float(np.interp(p.omega2, sec_w_sorted, sec_c_sorted))
            b2 = float(np.interp(p.omega2, sec_w_sorted, sec_b_sorted))
        else:
            mismatch = (p.omega2 - w_max) / p.omega2
            if mismatch > mismatch_tol:
                continue  # beyond the secondary's Roche limit
            c2 = float(sec_c_sorted[-1])
            b2 = float(sec_b_sorted[-1])
            omega2 = 0.5 * (p.omega2 + w_max)
        # equal densities: V2 = q V1  =>  s^3 * (b2 c2) = q * (b1 c1)
        s = (q * p.b * p.c / (b2 * c2)) ** (1.0 / 3.0)
        # Kepler III, point masses: omega^2 = G (m1 + m2) / d^3
        # => d^3 = (4/3) (1 + q) * (b1 c1) / [omega^2/(pi G rho)]
        d = ((4.0 / 3.0) * (1.0 + q) * p.b * p.c / omega2) ** (1.0 / 3.0)
        solutions.append(BinarySolution(
            q=q,
            primary_axes=np.array([1.0, p.b, p.c]),
            secondary_axes=s * np.array([1.0, b2, c2]),
            omega2=float(omega2),
            separation=float(d / (1.0 + s)),
            mismatch=float(mismatch),
        ))
    return solutions


__all__ = [
    "RocheComponent",
    "BinarySolution",
    "roche_component",
    "component_sequence",
    "match_binary",
]
