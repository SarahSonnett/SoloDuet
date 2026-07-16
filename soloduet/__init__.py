"""SoloDuet — is that light curve a solo elongated asteroid or a duet?

Fits both a single rotating triaxial ellipsoid (fluid-equilibrium Jacobi
figure by default, optionally unconstrained) and a tidally locked Roche
contact/close-binary model to a rotational light curve, then evaluates which
model the data statistically prefer, following Lacerda & Jewitt (2007,
AJ 133, 1393).

Quick start
-----------
>>> import numpy as np
>>> from soloduet import phase_fold, fit_lightcurve
>>> lc = phase_fold(time_mjd, mag, merr, period_hr=13.7744, alpha_deg=1.0)
>>> result = fit_lightcurve(lc)
>>> print(result.summary())

Conventions: periods in HOURS; differential magnitudes about the weighted
mean, increasing downward; one light-curve period = one rotation (solo)
= one orbit (duet).
"""

from ._compat import (HAVE_SILHOUETTE, HAVE_SPINDOC,  # noqa: F401
                      HAVE_SPOTLIGHT)
from .compare import MorphologyReport, morphology  # noqa: F401
from .fit import (ModelFit, SoloDuetResult, Verdict,  # noqa: F401
                  fit_lightcurve)
from .geometry import aspect_angle, view_vectors  # noqa: F401
from .io import FoldedLightcurve, phase_fold, read_lightcurve  # noqa: F401
from .jacobi import (analytic_dmag, density_from_spin,  # noqa: F401
                     jacobi_figure, jacobi_sequence)
from .render import (Body, binary_bodies, lightcurve,  # noqa: F401
                     render, solo_bodies)
from .roche import BinarySolution, match_binary, roche_component  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "fit_lightcurve",
    "SoloDuetResult",
    "ModelFit",
    "Verdict",
    "FoldedLightcurve",
    "phase_fold",
    "read_lightcurve",
    "morphology",
    "MorphologyReport",
    "jacobi_figure",
    "jacobi_sequence",
    "density_from_spin",
    "analytic_dmag",
    "roche_component",
    "match_binary",
    "BinarySolution",
    "render",
    "lightcurve",
    "Body",
    "solo_bodies",
    "binary_bodies",
    "view_vectors",
    "aspect_angle",
    "HAVE_SPOTLIGHT",
    "HAVE_SPINDOC",
    "HAVE_SILHOUETTE",
    "__version__",
]
