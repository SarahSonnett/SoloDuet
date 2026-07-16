"""Viewing geometry for light-curve synthesis.

SoloDuet adopts SpotLight's angle conventions so the two renderers can be
cross-validated directly:

* The body (or, for a binary, the co-rotating orbit) frame is fixed; rotation
  is implemented by advancing the observer's **west longitude**,
  ``wlong = 360 deg * phase``.
* A direction at latitude ``lat`` and west longitude ``w`` is the unit vector
  ``[cos(w) cos(lat), -sin(w) cos(lat), sin(lat)]``.
* The **aspect angle** ``theta`` is the angle between the line of sight and
  the spin (orbital) axis: sub-observer latitude ``= 90 - theta``.
* The Sun is placed at the same latitude as the observer with a *delta* west
  longitude equal to the solar phase angle ``alpha`` — exact for equator-on
  geometry and second-order at the small phase angles where light-curve
  model comparison is meaningful (Lacerda & Jewitt 2007 use alpha = 1 deg
  for KBOs).
"""

from __future__ import annotations

import numpy as np

RPD = np.pi / 180.0


def latlong_to_vector(lat_deg: float, wlong_deg: float) -> np.ndarray:
    """Unit vector for (latitude, west longitude), SpotLight convention."""
    lat = lat_deg * RPD
    w = wlong_deg * RPD
    return np.array([
        np.cos(w) * np.cos(lat),
        -np.sin(w) * np.cos(lat),
        np.sin(lat),
    ])


def view_vectors(aspect_deg: float, alpha_deg: float, phase: float):
    """Observer and Sun unit vectors at rotational phase ``phase`` in [0, 1).

    Returns ``(obs_dir, sun_dir)`` in the body/orbit frame.
    """
    lat = 90.0 - aspect_deg
    wlong = 360.0 * phase
    obs = latlong_to_vector(lat, wlong)
    sun = latlong_to_vector(lat, wlong + alpha_deg)
    return obs, sun


def aspect_angle(ecl_lon: float, ecl_lat: float,
                 pole_lon: float, pole_lat: float) -> float:
    """Aspect angle [deg] between the line of sight and a spin pole.

    All inputs are observer-centric ecliptic longitudes/latitudes in degrees
    (target direction and pole direction).
    """
    lam, bet = ecl_lon * RPD, ecl_lat * RPD
    lam_p, bet_p = pole_lon * RPD, pole_lat * RPD
    cos_theta = (np.sin(bet) * np.sin(bet_p)
                 + np.cos(bet) * np.cos(bet_p) * np.cos(lam - lam_p))
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def fetch_horizons_geometry(target: str, mjd_epochs, location: str = "500@399"):
    """Observer-centric ecliptic lon/lat and phase angle from JPL Horizons.

    Requires ``astroquery`` (optional dependency).  Returns a dict with
    arrays ``ecl_lon, ecl_lat, alpha, rhelio, delta``.
    """
    from astroquery.jplhorizons import Horizons  # deferred optional import

    mjd_epochs = np.atleast_1d(np.asarray(mjd_epochs, dtype=float))
    obj = Horizons(id=target, location=location, epochs=mjd_epochs + 2_400_000.5)
    eph = obj.ephemerides()
    return {
        "ecl_lon": np.array(eph["ObsEclLon"], dtype=float),
        "ecl_lat": np.array(eph["ObsEclLat"], dtype=float),
        "alpha": np.array(eph["alpha"], dtype=float),
        "rhelio": np.array(eph["r"], dtype=float),
        "delta": np.array(eph["delta"], dtype=float),
    }


__all__ = [
    "latlong_to_vector",
    "view_vectors",
    "aspect_angle",
    "fetch_horizons_geometry",
]
