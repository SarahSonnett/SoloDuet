"""Statistics and morphology diagnostics."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soloduet.compare import (aic, bic, bic_strength, chi2_with_zeropoint,
                              interp_periodic, morphology, scan_phi0)
from soloduet.io import FoldedLightcurve


def _lc(phase, dmag, err=0.02, alpha=1.0):
    e = np.full_like(dmag, err)
    return FoldedLightcurve(phase=phase, dmag=dmag, err=e, period_hr=10.0,
                            alpha_deg=alpha)


def test_interp_periodic_wraps():
    xp = np.arange(8) / 8.0
    fp = np.sin(2 * np.pi * xp)
    out = interp_periodic(np.array([-0.125, 1.125]), xp, fp)
    assert out[0] == pytest.approx(fp[-1])
    assert out[1] == pytest.approx(fp[1])


def test_chi2_zero_for_exact_model():
    ph = np.linspace(0, 1, 50, endpoint=False)
    model = 0.3 * np.sin(4 * np.pi * ph)
    lc = _lc(ph, model + 0.123)  # constant offset absorbed by zero-point
    chi2, zp = chi2_with_zeropoint(lc, ph, model, 0.0)
    assert chi2 < 1e-18
    assert zp == pytest.approx(0.123)


def test_scan_phi0_recovers_shift():
    ph = np.linspace(0, 1, 200, endpoint=False)
    model = 0.3 * np.sin(4 * np.pi * ph) + 0.1 * np.sin(2 * np.pi * ph)
    data = interp_periodic(ph - 0.2, ph, model)
    lc = _lc(ph, data)
    chi2, phi0, _ = scan_phi0(lc, ph, model)
    assert phi0 == pytest.approx(0.2, abs=1e-3)


def test_information_criteria():
    assert aic(100.0, 4, 50) == pytest.approx(108.0)
    assert bic(100.0, 4, 50) == pytest.approx(100.0 + 4 * np.log(50))
    assert bic_strength(1.0) == "indistinguishable"
    assert bic_strength(4.0) == "positive"
    assert bic_strength(8.0) == "strong"
    assert bic_strength(20.0) == "very strong"


def test_chi2_evidence_ladder():
    from soloduet.compare import chi2_evidence

    # real cases from this repo's field tests / survey work
    assert chi2_evidence(7.25, 7.36) == "indistinguishable"   # (16152)
    assert chi2_evidence(31.7, 33.3) == "indistinguishable"   # (4230)
    # a hair's-breadth adequacy straddle must NOT earn moderate: near-equal
    # chi^2_nu values support no claim regardless of the 3.0 threshold
    assert chi2_evidence(2.76, 3.45) == "indistinguishable"   # (222861)
    assert chi2_evidence(4.8, 8.3) == "weak"                  # (433) Eros
    assert chi2_evidence(2.8, 6.9) == "moderate"              # (216) Kleopatra
    assert chi2_evidence(0.94, 1.90) == "moderate"            # 2001 QG298
    # ladder thresholds
    assert chi2_evidence(2.0, 6.5) == "strong"                # ratio 3.25
    assert chi2_evidence(1.0, 5.5) == "very strong"           # ratio 5.5, adequate
    assert chi2_evidence(4.0, 21.0) == "strong"               # ratio > 5 but winner poor


def test_morphology_sinusoid_vs_notched():
    ph = np.linspace(0, 1, 300, endpoint=False)
    # pure sinusoid (ellipsoid-like): curvature ratio ~ 1, small odd power
    m_sin = 0.2 * -np.cos(4 * np.pi * ph)
    rep = morphology(_lc(ph, m_sin))
    assert not rep.exceeds_single_limit
    assert rep.minima_curvature_ratio == pytest.approx(1.0, abs=0.3)
    # eclipse-like curve: V-shaped faint minima (dmag maxima), rounded
    # bright maxima — the contact-binary signature
    m_v = 0.55 * (1.0 - np.abs(np.sin(2 * np.pi * ph)))
    rep_v = morphology(_lc(ph, m_v))
    assert rep_v.minima_curvature_ratio > 1.5


def test_morphology_amplitude_limit():
    ph = np.linspace(0, 1, 300, endpoint=False)
    rep = morphology(_lc(ph, 0.6 * -np.cos(4 * np.pi * ph)))  # 1.2 mag range
    assert rep.exceeds_single_limit
    assert any("0.9 mag" in n for n in rep.notes)


def test_morphology_alpha_flag():
    ph = np.linspace(0, 1, 100, endpoint=False)
    rep = morphology(_lc(ph, 0.2 * np.sin(4 * np.pi * ph), alpha=20.0))
    assert not rep.low_alpha_valid
