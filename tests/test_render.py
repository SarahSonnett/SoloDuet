"""Renderer validation: analytic curve, convergence, occultation, shadows."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soloduet.geometry import view_vectors
from soloduet.jacobi import analytic_dmag
from soloduet.render import binary_bodies, lightcurve, render, solo_bodies

# the module object (soloduet re-exports the render *function* under the
# same name, so a plain ``import soloduet.render as rd`` grabs the function)
rd = sys.modules["soloduet.render"]


def test_sphere_flux():
    # Lommel-Seeliger radiance is exactly 1/2 at alpha = 0, so the flux of a
    # unit sphere is pi/2
    obs, sun = view_vectors(90.0, 0.0, 0.0)
    f = render(solo_bodies((1, 1, 1)), obs, sun, n_pixels=256,
               scattering="lunar")
    assert f == pytest.approx(np.pi / 2, rel=2e-3)


def test_sphere_curve_flat():
    ph, dm = lightcurve(solo_bodies((1, 1, 1)), 75.0, 3.0, n_phases=16,
                        n_pixels=128)
    assert np.abs(dm).max() < 1e-3


def test_ls_matches_analytic():
    # equator-on, alpha = 0: LS flux is proportional to the projected area,
    # so the curve must match Lacerda & Jewitt (2007) eq. 3 (the ~0.1% gate)
    ph, dm = lightcurve(solo_bodies((1.0, 2 / 3, 0.5)), 90.0, 0.0,
                        n_phases=32, n_pixels=192, scattering="lunar")
    dm = dm - dm.mean()
    ana = analytic_dmag(2 / 3, ph)
    ana = ana - ana.mean()
    assert np.abs(dm - ana).max() < 2.5e-3  # mag  (~0.23% flux)


def test_resolution_convergence():
    errs = []
    for npix in (64, 128, 256):
        ph, dm = lightcurve(solo_bodies((1.0, 2 / 3, 0.5)), 90.0, 0.0,
                            n_phases=8, n_pixels=npix, scattering="lunar")
        dm = dm - dm.mean()
        ana = analytic_dmag(2 / 3, ph)
        ana = ana - ana.mean()
        errs.append(np.abs(dm - ana).max())
    assert errs[2] < errs[0]


def test_occultation_equal_spheres():
    # two touching unit spheres: at conjunction one is fully hidden, so the
    # flux halves relative to the side view (0.753 mag)
    bods = binary_bodies((1, 1, 1), (1, 1, 1), 2.0)
    obs, sun = view_vectors(90.0, 0.0, 0.0)
    f_conj = render(bods, obs, sun, n_pixels=192)
    obs, sun = view_vectors(90.0, 0.0, 0.25)
    f_side = render(bods, obs, sun, n_pixels=192)
    assert f_conj / f_side == pytest.approx(0.5, rel=5e-3)


def test_shadowing_reduces_flux(monkeypatch):
    # side-by-side touching spheres at alpha = 30 deg: the companion's
    # shadow must remove flux compared to a shadow-disabled render
    bods = binary_bodies((1, 1, 1), (1, 1, 1), 2.0)
    obs, sun = view_vectors(90.0, 30.0, 0.25)
    f_shadow = render(bods, obs, sun, n_pixels=192)
    monkeypatch.setattr(rd, "SHADOW_ALPHA_MIN", 1e9)  # disable shadow pass
    f_noshadow = render(bods, obs, sun, n_pixels=192)
    assert f_shadow < f_noshadow


def test_binary_deeper_minimum_than_ellipsoid():
    # a near-contact equal binary shows a deeper, sharper minimum than any
    # single Jacobi figure of the same amplitude class
    bods = binary_bodies((1, 0.8, 0.75), (1, 0.8, 0.75), 2.05)
    ph, dm = lightcurve(bods, 90.0, 1.0, n_phases=64, n_pixels=128)
    assert dm.max() - dm.min() > 0.9  # exceeds the single-figure limit
