"""Jacobi sequence: bifurcation anchor, stability limit, density scaling."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soloduet.jacobi import (BA_MIN, analytic_dmag, density_from_spin,
                             jacobi_figure, jacobi_sequence)


def test_bifurcation_anchor():
    # Maclaurin-Jacobi bifurcation (Chandrasekhar 1969)
    fig = jacobi_figure(1.0)
    assert abs(fig.c_over_a - 0.582724) < 1e-5
    assert abs(fig.omega2 - 0.374230) < 1e-5


def test_fission_limit_enforced():
    with pytest.raises(ValueError):
        jacobi_figure(0.30)


def test_sequence_monotonic():
    seq = jacobi_sequence(50)
    # flatter and slower toward higher elongation
    assert np.all(np.diff(seq["c_over_a"]) > 0)
    assert np.all(np.diff(seq["omega2"]) > 0)
    assert seq["b_over_a"][0] == pytest.approx(BA_MIN)


def test_density_scaling():
    # rho scales as 1/(P^2 omega2); anchor: omega2 = 0.130 at P = 13.7744 h
    # gives ~590 kg/m3 (2001 QG298, Lacerda & Jewitt 2007 Table 5)
    rho = density_from_spin(0.130, 13.7744)
    assert abs(rho - 589.0) < 5.0
    assert density_from_spin(0.130, 13.7744 / 2) == pytest.approx(4 * rho, rel=1e-9)


def test_analytic_dmag_amplitude():
    ph = np.linspace(0, 1, 400, endpoint=False)
    dm = analytic_dmag(2.0 / 3.0, ph)
    assert dm.max() - dm.min() == pytest.approx(2.5 * np.log10(1.5), abs=1e-6)
    # faintest at phase 0 (long axis end-on), brightest at 0.25
    assert dm[0] == pytest.approx(dm.max(), abs=1e-9)
    assert dm[100] == pytest.approx(dm.min(), abs=1e-9)
