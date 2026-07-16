"""Cached forward light-curve models.

Every distinct combination of (shape, aspect angle, scattering law) is
rendered **once** onto a dense phase grid and memoized; the fit layer then
evaluates rotational phase offsets and magnitude zero-points against the
cached curve at no rendering cost — the performance keystone of the grid
search.

Shape sources:

* ``jacobi_axes(b_over_a)`` — interpolated on the precomputed Jacobi
  sequence (:mod:`soloduet.tables`).
* free triaxial axes ``(1, b/a, c/a)`` — unconstrained mode.
* ``binary_scene(q, c1)`` — interpolated within the fixed-``q`` family of
  the precomputed Roche-binary grid.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

from . import tables
from .render import binary_bodies, lightcurve, solo_bodies

#: defaults chosen from the renderer convergence test (docs: ~0.16% flux)
DEFAULT_N_PIXELS = 192
DEFAULT_N_PHASES = 128

#: rounding applied to cache keys (renders are insensitive below this)
_KEY_DECIMALS = 4


def jacobi_axes(b_over_a: float) -> Tuple[float, float, float]:
    """Semi-axes ``(1, b/a, c/a)`` on the Jacobi sequence (table interpolation)."""
    t = tables.load_jacobi()
    b = float(np.clip(b_over_a, t["b_over_a"][0], t["b_over_a"][-1]))
    c = float(np.interp(b, t["b_over_a"], t["c_over_a"]))
    return (1.0, b, c)


def jacobi_omega2(b_over_a: float) -> float:
    """Normalized spin ``omega^2/(pi G rho)`` on the Jacobi sequence."""
    t = tables.load_jacobi()
    b = float(np.clip(b_over_a, t["b_over_a"][0], t["b_over_a"][-1]))
    return float(np.interp(b, t["b_over_a"], t["omega2"]))


def binary_families() -> dict:
    """Roche grid regrouped as ``{q: dict-of-arrays sorted by c1}``."""
    return _binary_families()


@lru_cache(maxsize=1)
def _binary_families() -> dict:
    t = tables.load_roche()
    fams: dict = {}
    for q in np.unique(t["q"]):
        sel = t["q"] == q
        order = np.argsort(t["c1"][sel])
        fams[float(q)] = {k: t[k][sel][order] for k in t if k != "q"}
    return fams


def binary_scene(q: float, c1: float):
    """Interpolated Roche-binary scene parameters at (``q``, primary ``c1``).

    Returns ``(primary_axes, secondary_axes, center_distance, omega2,
    separation)`` with the secondary axes already physically rescaled.
    ``q`` snaps to the nearest grid family; ``c1`` interpolates within it.
    """
    fams = _binary_families()
    q_grid = np.array(sorted(fams))
    qn = float(q_grid[np.argmin(np.abs(q_grid - q))])
    f = fams[qn]
    c1 = float(np.clip(c1, f["c1"][0], f["c1"][-1]))

    def interp(key):
        return float(np.interp(c1, f["c1"], f[key]))

    prim = np.array([1.0, interp("b1"), c1])
    sec = np.array([interp("a2"), interp("b2"), interp("c2")])
    separation = interp("separation")
    omega2 = interp("omega2")
    d = separation * (1.0 + sec[0])
    return prim, sec, d, omega2, separation


def _key(x: float) -> float:
    return round(float(x), _KEY_DECIMALS)


@lru_cache(maxsize=8192)
def _cached_lightcurve(scene_key, aspect_deg, alpha_deg, scattering,
                       n_phases, n_pixels):
    """Render (once) the dense-phase-grid curve for a hashable scene key."""
    kind = scene_key[0]
    if kind == "solo":
        bodies = solo_bodies(scene_key[1])
    else:
        _, prim, sec, dist = scene_key
        bodies = binary_bodies(prim, sec, dist)
    phases, dmag = lightcurve(bodies, aspect_deg, alpha_deg,
                              n_phases=n_phases, n_pixels=n_pixels,
                              scattering=scattering)
    return phases, dmag


def solo_curve(axes, aspect_deg: float, alpha_deg: float, scattering: str,
               n_phases: int = DEFAULT_N_PHASES,
               n_pixels: int = DEFAULT_N_PIXELS):
    """Cached dense light curve of a single ellipsoid."""
    key = ("solo", tuple(_key(x) for x in axes))
    return _cached_lightcurve(key, _key(aspect_deg), _key(alpha_deg),
                              str(scattering), int(n_phases), int(n_pixels))


def duet_curve(q: float, c1: float, aspect_deg: float, alpha_deg: float,
               scattering: str,
               n_phases: int = DEFAULT_N_PHASES,
               n_pixels: int = DEFAULT_N_PIXELS):
    """Cached dense light curve of a Roche binary at (``q``, ``c1``)."""
    prim, sec, dist, _, _ = binary_scene(q, c1)
    key = ("duet",
           tuple(_key(x) for x in prim),
           tuple(_key(x) for x in sec),
           _key(dist))
    return _cached_lightcurve(key, _key(aspect_deg), _key(alpha_deg),
                              str(scattering), int(n_phases), int(n_pixels))


def clear_cache() -> None:
    """Drop all memoized renders (frees memory between unrelated fits)."""
    _cached_lightcurve.cache_clear()


__all__ = [
    "jacobi_axes",
    "jacobi_omega2",
    "binary_scene",
    "binary_families",
    "solo_curve",
    "duet_curve",
    "clear_cache",
    "DEFAULT_N_PIXELS",
    "DEFAULT_N_PHASES",
]
