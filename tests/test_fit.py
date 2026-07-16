"""End-to-end round trips: synthesize, fit, recover the right verdict."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soloduet as sd
from soloduet.compare import interp_periodic
from soloduet.models import duet_curve, jacobi_axes, solo_curve

FAST_KW = dict(aspect_grid=(90.0,), scatterings=("lunar",), ba_step=0.1,
               q_step=0.2, n_c1=4, grid_n_pixels=64, grid_n_phases=32,
               n_pixels=96, n_phases=64)


def _sample(model_phase, model_dmag, phi0, noise, n, rng):
    ph = np.sort(rng.uniform(0, 1, n))
    dm = interp_periodic(ph - phi0, model_phase, model_dmag)
    return ph, dm + rng.normal(0, noise, n)


def test_duet_roundtrip():
    rng = np.random.default_rng(7)
    mp, md = duet_curve(0.80, 0.60, 90.0, 1.0, "lunar",
                        n_phases=96, n_pixels=128)
    ph, dm = _sample(mp, md, 0.13, 0.03, 80, rng)
    lc = sd.FoldedLightcurve(phase=ph, dmag=dm, err=np.full(80, 0.03),
                             period_hr=13.7744, alpha_deg=1.0)
    res = sd.fit_lightcurve(lc, **FAST_KW)
    assert res.verdict.preferred == "binary"
    b = res.fits["binary"]
    assert b.params["q"] == pytest.approx(0.80, abs=0.2)
    assert b.redchi2 < 2.0
    # density within ~15% of the injected figure's
    from soloduet.models import binary_scene
    from soloduet.jacobi import density_from_spin
    _, _, _, w2_true, _ = binary_scene(0.80, 0.60)
    rho_true = density_from_spin(w2_true, 13.7744)
    assert b.density_kgm3 == pytest.approx(rho_true, rel=0.15)


def test_solo_roundtrip():
    rng = np.random.default_rng(11)
    axes = jacobi_axes(0.60)
    mp, md = solo_curve(axes, 90.0, 1.0, "lunar", n_phases=96, n_pixels=128)
    ph, dm = _sample(mp, md, 0.40, 0.02, 80, rng)
    lc = sd.FoldedLightcurve(phase=ph, dmag=dm, err=np.full(80, 0.02),
                             period_hr=7.0, alpha_deg=1.0)
    res = sd.fit_lightcurve(lc, **FAST_KW)
    assert res.verdict.preferred == "single"
    j = res.fits["jacobi"]
    assert j.params["b_over_a"] == pytest.approx(0.60, abs=0.06)
    assert j.redchi2 < 2.0
    assert j.density_kgm3 is not None


def test_noise_indeterminate():
    # featureless noise: neither model should win decisively
    rng = np.random.default_rng(3)
    ph = np.sort(rng.uniform(0, 1, 60))
    dm = rng.normal(0, 0.05, 60)
    lc = sd.FoldedLightcurve(phase=ph, dmag=dm, err=np.full(60, 0.05),
                             period_hr=10.0, alpha_deg=1.0)
    res = sd.fit_lightcurve(lc, refine=False, **FAST_KW)
    assert res.verdict.preferred == "indeterminate"


def test_free_mode_has_no_density():
    rng = np.random.default_rng(5)
    mp, md = solo_curve((1.0, 0.6, 0.45), 90.0, 1.0, "lunar",
                        n_phases=64, n_pixels=96)
    ph, dm = _sample(mp, md, 0.0, 0.03, 50, rng)
    lc = sd.FoldedLightcurve(phase=ph, dmag=dm, err=np.full(50, 0.03),
                             period_hr=6.0, alpha_deg=1.0)
    res = sd.fit_lightcurve(lc, modes=("free",), refine=False, **FAST_KW)
    f = res.fits["free"]
    assert f.density_kgm3 is None
    assert any("no density" in n for n in f.notes)


def test_summary_renders():
    rng = np.random.default_rng(9)
    mp, md = solo_curve(jacobi_axes(0.7), 90.0, 1.0, "lunar",
                        n_phases=64, n_pixels=96)
    ph, dm = _sample(mp, md, 0.0, 0.03, 40, rng)
    lc = sd.FoldedLightcurve(phase=ph, dmag=dm, err=np.full(40, 0.03),
                             period_hr=8.0, alpha_deg=1.0,
                             object_name="synthetic")
    res = sd.fit_lightcurve(lc, refine=False, **FAST_KW)
    text = res.summary()
    assert "verdict" in text and "synthetic" in text


def test_verdict_error_scaling_16152_case():
    # Both models fit poorly (chi2_nu ~ 7) and differ by a whisker: without
    # error scaling, n = 434 would inflate that into "very strong" evidence;
    # with the Lacerda & Jewitt (2007) rescaling the verdict must be
    # indeterminate (the real (16152) 2019 situation).
    import soloduet.fit as sdfit

    rng = np.random.default_rng(1)
    n = 434
    ph = np.sort(rng.uniform(0, 1, n))
    lc = sd.FoldedLightcurve(phase=ph, dmag=0.2 * np.sin(4 * np.pi * ph),
                             err=np.full(n, 0.02), period_hr=22.93,
                             alpha_deg=1.0)
    curve = (np.arange(64) / 64.0, np.zeros(64))
    single = sdfit.ModelFit(name="jacobi", scattering="lunar",
                            params={"phi0": 0.0, "zeropoint": 0.0},
                            chi2=7.36 * (n - 4), n=n, k=4, curve=curve)
    binary = sdfit.ModelFit(name="binary", scattering="lunar",
                            params={"phi0": 0.0, "zeropoint": 0.0,
                                    "q": 0.6, "c1": 0.7, "separation": 1.0},
                            chi2=7.25 * (n - 5), n=n, k=5, curve=curve)
    v = sdfit._verdict(lc, {"jacobi": single, "binary": binary})
    assert v.chi2_scale > 5.0
    assert abs(v.delta_bic) < 6.0          # no longer "very strong"
    assert v.preferred == "indeterminate"
    # sanity: with trusted errors the same chi2 gap would be decisive
    raw = single.bic - binary.bic
    assert raw > 30.0
