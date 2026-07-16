"""Chandrasekhar index symbols for homogeneous triaxial ellipsoids.

For an ellipsoid with semi-axes ``a >= b >= c`` the dimensionless index
symbols (Chandrasekhar & Lebovitz 1962; Chandrasekhar 1969, Ch. 3, "Ellipsoidal
Figures of Equilibrium") are

    A_i = a b c * Integral_0^inf du / (Delta(u) * (a_i^2 + u)),
    Delta(u) = sqrt((a^2+u)(b^2+u)(c^2+u)),

with the sum rule ``A_1 + A_2 + A_3 = 2`` (sphere: 2/3 each).

Implementation notes
--------------------
The single-factor symbols are evaluated with Carlson's symmetric elliptic
integral of the second kind, R_D (``scipy.special.elliprd``):

    A_i = (2/3) a b c R_D(a_j^2, a_k^2, a_i^2),

which is numerically stable for *all* axis ratios, including the degenerate
oblate/prolate/spherical limits, avoiding the catastrophic cancellation that
affects the Legendre-form expressions when a ~ b or b ~ c.

The double-factor symbol

    A_ij = a b c * Integral_0^inf du / (Delta(u) * (a_i^2+u)(a_j^2+u))

is computed by the partial-fraction identity ``A_ij = (A_i - A_j)/(a_j^2 -
a_i^2)`` away from degeneracy and by direct adaptive quadrature when
``a_i ~ a_j``.  Quadrature versions of everything are provided for testing.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.special import elliprd

#: relative axis difference below which A_ij falls back to quadrature
_DEGENERATE_RTOL = 1e-5


def index_symbols(a: float, b: float, c: float):
    """Return ``(A1, A2, A3)`` for semi-axes ``a >= b >= c > 0``.

    Uses Carlson R_D; exact in the spherical/oblate/prolate limits.
    """
    a2, b2, c2 = a * a, b * b, c * c
    abc = a * b * c
    A1 = (2.0 / 3.0) * abc * elliprd(b2, c2, a2)
    A2 = (2.0 / 3.0) * abc * elliprd(a2, c2, b2)
    A3 = (2.0 / 3.0) * abc * elliprd(a2, b2, c2)
    return float(A1), float(A2), float(A3)


def _quad_single(a: float, b: float, c: float, axis: float) -> float:
    """A_i by adaptive quadrature (testing / cross-check)."""
    a2, b2, c2 = a * a, b * b, c * c
    p2 = axis * axis

    def integrand(u):
        delta = np.sqrt((a2 + u) * (b2 + u) * (c2 + u))
        return 1.0 / (delta * (p2 + u))

    val, _ = integrate.quad(integrand, 0.0, np.inf, epsabs=1e-12, epsrel=1e-11)
    return a * b * c * val


def index_symbols_quad(a: float, b: float, c: float):
    """``(A1, A2, A3)`` by quadrature — slow reference implementation."""
    return (
        _quad_single(a, b, c, a),
        _quad_single(a, b, c, b),
        _quad_single(a, b, c, c),
    )


def _quad_double(a: float, b: float, c: float, ai: float, aj: float) -> float:
    """A_ij by adaptive quadrature."""
    a2, b2, c2 = a * a, b * b, c * c
    p2, q2 = ai * ai, aj * aj

    def integrand(u):
        delta = np.sqrt((a2 + u) * (b2 + u) * (c2 + u))
        return 1.0 / (delta * (p2 + u) * (q2 + u))

    val, _ = integrate.quad(integrand, 0.0, np.inf, epsabs=1e-12, epsrel=1e-11)
    return a * b * c * val


def double_symbol(a: float, b: float, c: float, ai: float, aj: float) -> float:
    """A_ij = abc * Int du / (Delta (a_i^2+u)(a_j^2+u)).

    Partial fractions where safe, quadrature near the ``a_i ~ a_j``
    degeneracy.
    """
    if abs(ai - aj) <= _DEGENERATE_RTOL * max(ai, aj):
        return _quad_double(a, b, c, ai, aj)
    Ai = (2.0 / 3.0) * a * b * c * elliprd(*_rd_args(a, b, c, ai))
    Aj = (2.0 / 3.0) * a * b * c * elliprd(*_rd_args(a, b, c, aj))
    return float((Ai - Aj) / (aj * aj - ai * ai))


def _rd_args(a: float, b: float, c: float, axis: float):
    """R_D argument triple (other1^2, other2^2, axis^2) for the given axis."""
    axes = [a, b, c]
    # remove one occurrence of `axis` (works when axes coincide in value)
    idx = min(range(3), key=lambda i: abs(axes[i] - axis))
    others = [axes[i] for i in range(3) if i != idx]
    return others[0] ** 2, others[1] ** 2, axis ** 2


__all__ = [
    "index_symbols",
    "index_symbols_quad",
    "double_symbol",
]
