"""NumPy ray-trace renderer for scenes of one or two triaxial ellipsoids.

Replaces the POV-Ray step of Lacerda & Jewitt (2007) with a self-contained,
fully vectorized orthographic ray tracer:

* Sun and observer at infinity; parallel rays on a pixel grid perpendicular
  to the line of sight.
* Analytic ray-ellipsoid intersection (unit-sphere transform, quadratic).
* A z-buffer across the bodies provides **mutual occultation** for free.
* **Mutual shadowing** is handled with analytic shadow rays toward the Sun
  against the companion body (skipped below ``alpha`` = 0.01 deg, where
  occultation alone is exact).
* Per-pixel incidence/emission cosines feed a scattering law
  (:mod:`soloduet.scattering`); the pixel sum gives the disk-integrated flux.

Bodies are axis-aligned in the scene frame (for a tidally locked binary the
long axes lie along the line of centers — the scene x-axis — and the spin
axes along z), so rotation is implemented by rotating the observer/Sun
vectors (:func:`soloduet.geometry.view_vectors`), never the geometry.

Validation: at zero phase angle the Lommel-Seeliger disk-integrated flux of
a single ellipsoid is exactly proportional to its projected area, so the
rendered light curve must match the analytic curve of Lacerda & Jewitt
(2007, eq. 3) — see ``tests/test_render.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .geometry import view_vectors
from .scattering import get_law

#: phase angle [deg] below which shadow rays are skipped (occultation exact)
SHADOW_ALPHA_MIN = 0.01


@dataclass
class Body:
    """An axis-aligned triaxial ellipsoid in the scene frame."""

    semi_axes: np.ndarray                       # (a, b, c)
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.semi_axes = np.asarray(self.semi_axes, dtype=float)
        self.center = np.asarray(self.center, dtype=float)


def solo_bodies(axes: Sequence[float]) -> List[Body]:
    """Scene for a single ellipsoid at the origin."""
    return [Body(np.asarray(axes, dtype=float))]


def binary_bodies(primary_axes: Sequence[float],
                  secondary_axes: Sequence[float],
                  center_distance: float) -> List[Body]:
    """Scene for a tidally locked binary.

    Components sit on the x-axis (the line of centers), separated by
    ``center_distance``, with the system barycenter (equal densities: masses
    proportional to axis products) at the origin.
    """
    p = np.asarray(primary_axes, dtype=float)
    s = np.asarray(secondary_axes, dtype=float)
    m1, m2 = np.prod(p), np.prod(s)
    x1 = -center_distance * m2 / (m1 + m2)
    x2 = center_distance * m1 / (m1 + m2)
    return [Body(p, np.array([x1, 0.0, 0.0])),
            Body(s, np.array([x2, 0.0, 0.0]))]


def _ray_hits(origin: np.ndarray, direction: np.ndarray, body: Body):
    """Intersection parameters of rays ``origin + s * direction`` with a body.

    ``origin`` is (N, 3); ``direction`` is (3,).  Returns ``(s_far, hit)``
    where ``s_far`` is the root closest to the ray start *from the far
    side* — i.e. the largest root — and ``hit`` is a boolean mask.
    """
    inv = 1.0 / body.semi_axes
    P = (origin - body.center) * inv          # (N, 3)
    D = direction * inv                       # (3,)
    a = float(np.dot(D, D))
    b = P @ D                                  # (N,)
    c = np.einsum("ij,ij->i", P, P) - 1.0
    disc = b * b - a * c
    hit = disc >= 0.0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    s_far = (-b + sq) / a
    return s_far, hit


def _shadowed(points: np.ndarray, sun_dir: np.ndarray, body: Body) -> np.ndarray:
    """True where a surface point is shadowed by ``body`` toward the Sun."""
    inv = 1.0 / body.semi_axes
    P = (points - body.center) * inv
    D = sun_dir * inv
    a = float(np.dot(D, D))
    b = P @ D
    c = np.einsum("ij,ij->i", P, P) - 1.0
    disc = b * b - a * c
    with np.errstate(invalid="ignore"):
        s_far = (-b + np.sqrt(np.where(disc >= 0.0, disc, 0.0))) / a
    # the ray enters the companion at some s > 0 (epsilon guards self-acne)
    return (disc > 0.0) & (s_far > 1e-9)


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3-vector cross product (avoids np.cross overhead in the hot path)."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def _image_basis(obs_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal (right, up) spanning the image plane, z-up convention."""
    right = _cross3(np.array([0.0, 0.0, 1.0]), obs_dir)
    if np.dot(right, right) < 1e-12:
        right = _cross3(obs_dir, np.array([1.0, 0.0, 0.0]))
    up = _cross3(obs_dir, right)
    up /= np.linalg.norm(up)
    right = _cross3(up, obs_dir)
    right /= np.linalg.norm(right)
    return right, up


def scene_radius(bodies: List[Body]) -> float:
    """Radius of a bounding sphere of the scene about the origin."""
    return max(float(np.linalg.norm(b.center) + b.semi_axes.max()) for b in bodies)


def render(
    bodies: List[Body],
    obs_dir: np.ndarray,
    sun_dir: np.ndarray,
    n_pixels: int = 192,
    scattering="lunar",
    return_image: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """Disk-integrated flux of the scene (optionally also the rendered image).

    Flux is in arbitrary but internally consistent units (brightness summed
    over pixels times pixel area), so flux *ratios* across rotational phases
    — hence differential magnitudes — are well defined.
    """
    law = get_law(scattering)
    obs_dir = np.asarray(obs_dir, dtype=float)
    sun_dir = np.asarray(sun_dir, dtype=float)
    alpha = float(np.arccos(np.clip(np.dot(obs_dir, sun_dir), -1.0, 1.0)))

    right, up = _image_basis(obs_dir)
    half = scene_radius(bodies)
    n = int(n_pixels)
    coords = (np.arange(n) + 0.5) / n * 2.0 * half - half
    uu, vv = np.meshgrid(coords, coords)
    origins = uu.ravel()[:, None] * right + vv.ravel()[:, None] * up  # (N, 3)
    pixel_area = (2.0 * half / n) ** 2

    # z-buffer across bodies: keep the intersection nearest the observer
    npix = origins.shape[0]
    best_s = np.full(npix, -np.inf)
    winner = np.full(npix, -1, dtype=int)
    for k, body in enumerate(bodies):
        s_far, hit = _ray_hits(origins, obs_dir, body)
        better = hit & (s_far > best_s)
        best_s[better] = s_far[better]
        winner[better] = k

    brights = np.zeros(npix)
    on_disk = winner >= 0
    if on_disk.any():
        idx = np.where(on_disk)[0]
        points = origins[idx] + best_s[idx, None] * obs_dir
        mu0 = np.zeros(idx.size)
        mu = np.zeros(idx.size)
        for k, body in enumerate(bodies):
            sel = winner[idx] == k
            if not sel.any():
                continue
            normals = (points[sel] - body.center) / body.semi_axes ** 2
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            mu[sel] = normals @ obs_dir
            mu0[sel] = normals @ sun_dir
        lit = (mu > 0.0) & (mu0 > 0.0)

        # mutual shadowing (only meaningful off zero phase angle)
        if len(bodies) > 1 and np.degrees(alpha) >= SHADOW_ALPHA_MIN:
            for k, body in enumerate(bodies):
                sel = lit & (winner[idx] == k)
                if not sel.any():
                    continue
                other = bodies[1 - k]
                sh = _shadowed(points[sel] + 1e-9 * obs_dir, sun_dir, other)
                tmp = lit[sel]
                tmp[sh] = False
                lit[sel] = tmp

        vals = np.zeros(idx.size)
        if lit.any():
            vals[lit] = law(mu0[lit], mu[lit], alpha, None)
        brights[idx] = np.maximum(vals, 0.0)

    flux = float(np.sum(brights) * pixel_area)
    if return_image:
        return flux, brights.reshape(n, n)
    return flux


def lightcurve(
    bodies: List[Body],
    aspect_deg: float,
    alpha_deg: float,
    n_phases: int = 128,
    n_pixels: int = 192,
    scattering="lunar",
    phases: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Differential-magnitude light curve over one rotation (or orbit).

    Returns ``(phases, dmag)`` with ``dmag`` relative to the mean flux;
    magnitudes increase downward (fainter = more positive).
    """
    if phases is None:
        phases = np.arange(int(n_phases)) / float(n_phases)
    phases = np.asarray(phases, dtype=float)
    flux = np.empty(phases.size)
    for i, ph in enumerate(phases):
        obs, sun = view_vectors(aspect_deg, alpha_deg, float(ph))
        flux[i] = render(bodies, obs, sun, n_pixels=n_pixels,
                         scattering=scattering)
    if np.any(flux <= 0.0):
        raise RuntimeError("Zero disk-integrated flux encountered; "
                           "increase n_pixels or check the geometry")
    dmag = -2.5 * np.log10(flux / flux.mean())
    return phases, dmag


__all__ = [
    "Body",
    "solo_bodies",
    "binary_bodies",
    "render",
    "lightcurve",
    "scene_radius",
]
