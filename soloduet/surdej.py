"""Analytic geometric-scattering light curve of a triaxial ellipsoid.

Surdej & Surdej (1978, A&A 66, 31) model the rotational brightness variation
of a three-axis ellipsoid whose flux is proportional to its projected area
("geometric" scattering).  For semi-axes ``a >= b >= c`` rotating about the
``c`` axis, viewed at aspect angle ``theta`` (pole vs. line of sight) and
rotation angle ``psi = 2 pi phase`` measured from the long axis, the
projected area is exactly

    S(psi) = pi sqrt( b^2 c^2 sin^2(theta) cos^2(psi)
                    + a^2 c^2 sin^2(theta) sin^2(psi)
                    + a^2 b^2 cos^2(theta) )

Special cases wired into the test suite:

* equator-on (``theta = 90``): reduces to eq. 3 of Lacerda & Jewitt (2007);
* prolate (``b = c``): ``S/S_max = sqrt(1 - k^2 cos^2 psi)`` with
  ``k^2 = (1 - (b/a)^2) sin^2(theta)`` — the form used in the Simmer survey
  simulator, whose peak-to-peak amplitude is the Sheppard & Jewitt (2004)
  amplitude-aspect relation.

Geometric scattering equals the zero-phase-angle limit of any physical law
that conserves the illuminated cross-section; the fit layer uses this
analytic curve for the solo models when the ``"geometric"`` law is selected
(valid for small solar phase angles) and the uniform-radiance render for the
binary, which has no closed form.
"""

from __future__ import annotations

import numpy as np


def projected_area(axes, aspect_deg: float, phase) -> np.ndarray:
    """Projected area of a triaxial ellipsoid over rotational phase.

    ``axes = (a, b, c)`` with rotation about ``c``; ``phase`` in [0, 1),
    zero when the long axis points at the observer (light-curve minimum),
    matching the renderer's phase convention.
    """
    a, b, c = (float(x) for x in axes)
    theta = np.radians(aspect_deg)
    psi = 2.0 * np.pi * np.asarray(phase, dtype=float)
    s2, c2 = np.sin(theta) ** 2, np.cos(theta) ** 2
    return np.pi * np.sqrt(
        b * b * c * c * s2 * np.cos(psi) ** 2
        + a * a * c * c * s2 * np.sin(psi) ** 2
        + a * a * b * b * c2)


def surdej_dmag(axes, aspect_deg: float, phases) -> np.ndarray:
    """Differential magnitudes (about the mean flux) of the S&S78 curve.

    Normalized by the mean projected area over the supplied phases, matching
    the renderer's mean-flux convention so the two are interchangeable in
    the fit layer.
    """
    S = projected_area(axes, aspect_deg, phases)
    return -2.5 * np.log10(S / S.mean())


def amplitude_mag(b_over_a: float, aspect_deg: float) -> float:
    """Peak-to-peak amplitude [mag] of a prolate (b = c) ellipsoid.

    The Sheppard & Jewitt (2004) amplitude-aspect relation,

        dm = 2.5 log10(a/b) - 1.25 log10[ ((a/b)^2 - 1) cos^2(theta) + 1 ],

    with ``theta`` the aspect angle (0 = pole-on, 90 = equator-on); the
    exact peak-to-peak range of :func:`projected_area` for ``b = c``.
    """
    ab = 1.0 / float(b_over_a)
    cos2 = np.cos(np.radians(aspect_deg)) ** 2
    return float(2.5 * np.log10(ab)
                 - 1.25 * np.log10((ab * ab - 1.0) * cos2 + 1.0))


__all__ = ["projected_area", "surdej_dmag", "amplitude_mag"]
