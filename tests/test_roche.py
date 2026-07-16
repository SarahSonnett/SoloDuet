"""Roche components and matched binaries vs. Lacerda & Jewitt (2007) Table 3."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soloduet.roche import match_binary, roche_component


def test_component_shapes_table3():
    # q = 0.25 primary at c = 0.83 -> b = 0.91674; secondary (1/q = 4) at
    # c = 0.48 -> b = 0.51426 (Lacerda & Jewitt 2007, Table 3, row 1)
    prim = roche_component(0.25, 0.83)
    assert len(prim) >= 1
    assert abs(prim[0].b - 0.91674) < 2e-5
    sec = roche_component(4.0, 0.48)
    assert abs(sec[0].b - 0.51426) < 2e-5


def test_omega2_pairing_table3():
    # the two components' normalized spins bracket the tabulated value and
    # their mean reproduces it (the paper pairs 0.01-step grid nodes)
    wp = roche_component(0.25, 0.83)[0].omega2
    ws = roche_component(4.0, 0.48)[0].omega2
    assert abs(0.5 * (wp + ws) - 0.10626) < 5e-5


def test_matched_binary_q025():
    sols = match_binary(0.25)
    assert len(sols) > 5
    # near-contact end reproduces Table 3 row 1 within the method tolerance
    s = min(sols, key=lambda s: abs(s.primary_axes[2] - 0.83))
    assert abs(s.primary_axes[1] - 0.91674) < 1e-3
    assert abs(s.omega2 - 0.10626) < 2e-3
    assert abs(s.separation - 1.19222) < 0.02
    # equal densities: secondary volume = q * primary volume
    v1 = np.prod(s.primary_axes)
    v2 = np.prod(s.secondary_axes)
    assert v2 / v1 == pytest.approx(0.25, rel=1e-6)


def test_q1_symmetric():
    sols = match_binary(1.0)
    assert len(sols) > 10
    for s in sols[::10]:
        assert np.allclose(s.primary_axes, s.secondary_axes, rtol=1e-9)


def test_far_pair_spins_down():
    # wider separations rotate more slowly (Kepler)
    sols = match_binary(0.5)
    seps = np.array([s.separation for s in sols])
    w = np.array([s.omega2 for s in sols])
    order = np.argsort(seps)
    assert np.all(np.diff(w[order]) < 0)


def test_shipped_table_regression():
    from soloduet.tables import load_roche

    t = load_roche()
    assert t["q"].size > 1000
    # every stored solution keeps the equal-density volume ratio
    v1 = t["b1"] * t["c1"]
    v2 = t["a2"] * t["b2"] * t["c2"]
    assert np.allclose(v2 / v1, t["q"], rtol=1e-6)
    # mismatch never exceeds the matching tolerance
    assert t["mismatch"].max() <= 0.02 + 1e-9
