"""Surdej & Surdej (1978) analytic geometric model: reductions and parity."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soloduet.jacobi import analytic_dmag
from soloduet.render import lightcurve, solo_bodies
from soloduet.surdej import amplitude_mag, projected_area, surdej_dmag

PH = np.linspace(0, 1, 256, endpoint=False)


def test_equator_on_reduces_to_lj07_eq3():
    # theta = 90: S&S78 must equal Lacerda & Jewitt (2007) eq. 3
    dm = surdej_dmag((1.0, 2 / 3, 0.5), 90.0, PH)
    ana = analytic_dmag(2 / 3, PH)
    assert np.allclose(dm - dm.mean(), ana - ana.mean(), atol=1e-12)


def test_prolate_reduces_to_simmer_form():
    # b = c: S/S_max = sqrt(1 - k^2 cos^2 psi), k^2 = (1 - b^2) sin^2(theta)
    b, theta = 0.55, 63.0
    S = projected_area((1.0, b, b), theta, PH)
    k2 = (1.0 - b * b) * np.sin(np.radians(theta)) ** 2
    expected = np.sqrt(1.0 - k2 * np.cos(2 * np.pi * PH) ** 2)
    assert np.allclose(S / S.max(), expected / expected.max(), atol=1e-12)


def test_amplitude_is_sheppard_jewitt():
    # peak-to-peak of the curve equals the S&J04 amplitude-aspect relation
    for b, theta in [(0.5, 90.0), (0.5, 60.0), (0.7, 75.0), (0.9, 30.0)]:
        dm = surdej_dmag((1.0, b, b), theta, PH)
        assert dm.max() - dm.min() == pytest.approx(
            amplitude_mag(b, theta), abs=1e-4)


def test_pole_on_flat():
    dm = surdej_dmag((1.0, 0.5, 0.4), 0.0, PH)
    assert np.abs(dm).max() < 1e-12


def test_matches_uniform_render_at_zero_alpha():
    # the ray tracer with the uniform ("geometric") law must reproduce the
    # analytic projected-area curve at alpha = 0, any aspect
    axes = (1.0, 0.62, 0.47)
    for theta in (90.0, 65.0):
        ph, dm_render = lightcurve(solo_bodies(axes), theta, 0.0,
                                   n_phases=16, n_pixels=256,
                                   scattering="geometric")
        dm_ana = surdej_dmag(axes, theta, ph)
        assert np.abs((dm_render - dm_render.mean())
                      - (dm_ana - dm_ana.mean())).max() < 3e-3


def test_solo_curve_fast_path():
    from soloduet.models import solo_curve

    ph, dm = solo_curve((1.0, 0.6, 0.5), 80.0, 1.0, "geometric",
                        n_phases=64, n_pixels=32)  # n_pixels ignored
    assert ph.size == 64
    assert np.allclose(dm, surdej_dmag((1.0, 0.6, 0.5), 80.0, ph))
