"""The Jacobi sequence of triaxial fluid-equilibrium figures.

A homogeneous, self-gravitating fluid rotating uniformly about its shortest
axis assumes a Maclaurin spheroid or, at higher angular momentum, a Jacobi
triaxial ellipsoid (Chandrasekhar 1969).  With semi-axes ``a >= b >= c``
(``a = 1`` throughout), the Jacobi condition

    a^2 b^2 A_12 = c^2 A_3

fixes ``c`` for every ``b``, and the normalized spin follows as

    omega^2 / (pi G rho) = 2 (A_1 - b^2 A_12)

(Chandrasekhar 1969; Lacerda & Jewitt 2007, eqs. 1-2).  Combined with an
observed rotation period this yields the bulk density.  Figures with
``b/a < 0.43`` are unstable to rotational fission (Jeans 1919), which bounds
the sequence used here.

Anchor values (Chandrasekhar 1969): at the Maclaurin-Jacobi bifurcation
``b/a = 1``, ``c/a = 0.582724`` and ``omega^2/(pi G rho) = 0.374230``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .chandrasekhar import double_symbol, index_symbols

#: gravitational constant [m^3 kg^-1 s^-2]
G_SI = 6.674e-11

#: Jeans (1919) rotational-fission stability limit on b/a
BA_MIN = 0.43

SECONDS_PER_HOUR = 3600.0


@dataclass
class JacobiFigure:
    """One point on the Jacobi sequence (semi-axis ``a = 1``)."""

    b_over_a: float
    c_over_a: float
    omega2: float  # omega^2 / (pi G rho), dimensionless

    @property
    def axes(self) -> np.ndarray:
        return np.array([1.0, self.b_over_a, self.c_over_a])


def _jacobi_residual(c: float, b: float) -> float:
    """Residual of the Jacobi condition at semi-axes (1, b, c)."""
    _, _, A3 = index_symbols(1.0, b, c)
    A12 = double_symbol(1.0, b, c, 1.0, b)
    return b * b * A12 - c * c * A3


def jacobi_figure(b_over_a: float) -> JacobiFigure:
    """Solve the Jacobi condition for ``c/a`` and spin at the given ``b/a``.

    Raises ``ValueError`` outside the stable range ``0.43 <= b/a <= 1``.
    """
    b = float(b_over_a)
    if not (BA_MIN - 1e-9 <= b <= 1.0 + 1e-12):
        raise ValueError(
            f"b/a = {b:.4f} outside the stable Jacobi range "
            f"[{BA_MIN}, 1.0] (Jeans 1919 fission limit)"
        )
    b = min(b, 1.0)
    c = brentq(_jacobi_residual, 1e-4, b - 1e-10, args=(b,), xtol=1e-12)
    A1, _, _ = index_symbols(1.0, b, c)
    A12 = double_symbol(1.0, b, c, 1.0, b)
    omega2 = 2.0 * (A1 - b * b * A12)
    return JacobiFigure(b_over_a=b, c_over_a=float(c), omega2=float(omega2))


def jacobi_sequence(n: int = 200) -> dict:
    """Tabulate the Jacobi sequence over ``b/a`` in [0.43, 1.00].

    Returns a dict of arrays ``{"b_over_a", "c_over_a", "omega2"}``.
    """
    b_grid = np.linspace(BA_MIN, 1.0, int(n))
    c_grid = np.empty_like(b_grid)
    w_grid = np.empty_like(b_grid)
    for i, b in enumerate(b_grid):
        fig = jacobi_figure(b)
        c_grid[i] = fig.c_over_a
        w_grid[i] = fig.omega2
    return {"b_over_a": b_grid, "c_over_a": c_grid, "omega2": w_grid}


def density_from_spin(omega2: float, period_hr: float) -> float:
    """Bulk density [kg m^-3] from normalized spin and period (HOURS).

    ``rho = omega^2 / (pi G * [omega^2/(pi G rho)])`` with
    ``omega = 2 pi / P``.
    """
    period_s = period_hr * SECONDS_PER_HOUR
    omega = 2.0 * np.pi / period_s
    return float(omega * omega / (np.pi * G_SI * omega2))


def analytic_dmag(b_over_a: float, phase: np.ndarray) -> np.ndarray:
    """Analytic light curve of an ellipsoid, equator-on at zero phase angle.

    Differential magnitude (fainter = positive) from the projected area,

    ``dmag = -2.5 log10 sqrt(1 + [(b/a)^2 - 1] cos^2(2 pi phase))``,

    the (sign-adjusted) eq. 3 of Lacerda & Jewitt 2007 — valid for brightness
    proportional to the projected area, which is exactly the Lommel-Seeliger
    disk-integrated behaviour at zero phase angle.  ``phase = 0`` views the
    long (``a``) axis end-on (projected area ``pi b c``, the light-curve
    minimum); ``dmag = 0`` at the maxima; the full amplitude is
    ``2.5 log10(a/b)``.
    """
    ba2 = float(b_over_a) ** 2
    return -2.5 * np.log10(
        np.sqrt(1.0 + (ba2 - 1.0) * np.cos(2.0 * np.pi * phase) ** 2))


__all__ = [
    "JacobiFigure",
    "jacobi_figure",
    "jacobi_sequence",
    "density_from_spin",
    "analytic_dmag",
    "BA_MIN",
    "G_SI",
]
