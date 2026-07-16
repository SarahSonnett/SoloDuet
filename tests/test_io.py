"""I/O helpers: H-G phase correction and epoch synchronization."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soloduet.io import hg_phase_correction, phase_fold


def test_hg_zero_at_opposition():
    assert hg_phase_correction(0.0, G=0.15) == pytest.approx(0.0, abs=1e-9)


def test_hg_monotone_dimming():
    alpha = np.array([0.0, 2.0, 5.0, 10.0, 20.0])
    corr = hg_phase_correction(alpha, G=0.15)
    assert np.all(np.diff(corr) > 0)          # fainter at larger alpha
    # opposition surge (~0.3 mag over the first few degrees) plus the
    # ~0.035 mag/deg linear regime: ~0.66 mag at alpha = 10 for G = 0.15
    assert 0.4 < corr[3] < 0.9


def test_hg_g_dependence():
    # lower G -> steeper phase darkening
    assert hg_phase_correction(10.0, G=0.05) > hg_phase_correction(10.0, G=0.4)


def test_hg_synchronizes_epochs():
    # two epochs of a flat (non-varying) object at different phase angles:
    # after correction the folded curve is flat again
    rng = np.random.default_rng(0)
    n = 40
    t1 = 58700.0 + np.sort(rng.uniform(0, 0.3, n))
    t2 = 58760.0 + np.sort(rng.uniform(0, 0.3, n))
    a1, a2 = 2.0, 9.0
    m1 = 18.0 + hg_phase_correction(a1, 0.15) * np.ones(n)
    m2 = 18.0 + hg_phase_correction(a2, 0.15) * np.ones(n)
    # uncorrected fold: bimodal, ~the correction difference apart
    raw = np.concatenate([m1, m2])
    gap = hg_phase_correction(a2, 0.15) - hg_phase_correction(a1, 0.15)
    assert raw.max() - raw.min() == pytest.approx(gap, abs=1e-12)
    # corrected fold: flat
    corr = np.concatenate([
        m1 - hg_phase_correction(np.full(n, a1), 0.15),
        m2 - hg_phase_correction(np.full(n, a2), 0.15)])
    lc = phase_fold(np.concatenate([t1, t2]), corr, np.full(2 * n, 0.02),
                    period_hr=7.0)
    assert np.abs(lc.dmag).max() < 1e-9
