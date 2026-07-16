"""Cross-validation of the SoloDuet renderer against SpotLight.

Runs only when the sibling SpotLight repo is importable.  The Lambert law is
shared verbatim (radiance = mu0); SpotLight's Lommel-Seeliger variant is a
per-surface-area form and is not compared directly (see
soloduet.scattering docstring).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soloduet._compat import HAVE_SPOTLIGHT
from soloduet.render import lightcurve, solo_bodies

pytestmark = pytest.mark.skipif(not HAVE_SPOTLIGHT,
                                reason="SpotLight sibling repo not available")


@pytest.mark.parametrize("aspect,alpha", [(90, 0), (90, 15), (60, 5), (75, 1)])
def test_lambert_agreement(aspect, alpha):
    from spotlight import spotlight_lightcurve

    axes = (1.0, 2 / 3, 0.5)
    res = spotlight_lightcurve(list(axes), 90 - aspect, alpha, 90 - aspect,
                               resolution=24, n_pixels=384,
                               phot_func="lambertian", dmag=True)
    sl = np.array([r.dib for r in res])
    ph, dm = lightcurve(solo_bodies(axes), aspect, alpha, n_phases=24,
                        n_pixels=384, scattering="icy")
    assert np.abs(dm - sl).max() < 1e-3  # mag
