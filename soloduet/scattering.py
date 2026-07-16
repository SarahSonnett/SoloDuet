"""Surface scattering laws.

SoloDuet follows Lacerda & Jewitt (2007): light curves are modelled with the
two simplest laws bracketing realistic asteroid surfaces —

* **Lommel-Seeliger** (``"lunar"``): single-scattering, appropriate for dark
  surfaces; radiance ``r = mu0 / (mu0 + mu)`` (Lacerda & Jewitt 2007,
  eq. 7).  At zero phase angle its disk-integrated brightness is exactly
  proportional to the projected area.
* **Lambert** (``"icy"``): perfectly diffuse multiple scattering,
  representative of bright icy surfaces; radiance ``r = mu0``.

The Hapke model is deliberately excluded: its many parameters cannot be
constrained by disk-integrated light curves (Lacerda & Jewitt 2007).

These functions return **radiance** (brightness per unit *projected* area),
the quantity an image-plane renderer sums over pixels.  Note the distinction
from forms written per unit *surface* area, which carry an extra factor of
``mu`` (e.g. ``spotlight.photfuncs.lommel_seeliger``); the function
signature ``f(mu0, mu, alpha, arg)`` is shared with SpotLight so laws remain
interchangeable when that factor is accounted for.
"""

from __future__ import annotations

import numpy as np


def lambertian(mu0, mu, alpha, arg=None):
    """Lambert law: I/F = mu0."""
    return np.asarray(mu0).copy()


def lommel_seeliger(mu0, mu, alpha, arg=None):
    """Lommel-Seeliger radiance: r = mu0 / (mu0 + mu), 0 where unlit."""
    mu0 = np.asarray(mu0)
    mu = np.asarray(mu)
    denom = mu0 + mu
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0.0, mu0 / denom, 0.0)


#: user-facing law names (Lacerda & Jewitt 2007 nomenclature)
SCATTERING = {
    "lunar": lommel_seeliger,
    "icy": lambertian,
}


def get_law(name_or_func):
    """Resolve a scattering law from a name in :data:`SCATTERING` or a callable."""
    if callable(name_or_func):
        return name_or_func
    try:
        return SCATTERING[str(name_or_func)]
    except KeyError:
        raise ValueError(
            f"Unknown scattering law {name_or_func!r}; choose from {sorted(SCATTERING)}"
        ) from None


__all__ = ["lambertian", "lommel_seeliger", "SCATTERING", "get_law"]
