"""Index-symbol numerics: sum rule, limits, and quadrature agreement."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soloduet.chandrasekhar import (double_symbol, index_symbols,
                                    index_symbols_quad)


def test_sphere():
    A = index_symbols(1.0, 1.0, 1.0)
    assert np.allclose(A, 2.0 / 3.0, atol=1e-12)


def test_sum_rule():
    for axes in [(1, 0.7, 0.4), (1, 0.99, 0.2), (1, 0.43, 0.34), (1, 1, 0.58)]:
        assert abs(sum(index_symbols(*axes)) - 2.0) < 1e-10


def test_ordering():
    # larger axis -> smaller index symbol
    A1, A2, A3 = index_symbols(1.0, 0.7, 0.4)
    assert A1 < A2 < A3


def test_quad_agreement():
    for axes in [(1, 0.7, 0.4), (1, 0.9, 0.85), (1, 0.5, 0.45)]:
        carlson = np.array(index_symbols(*axes))
        quad = np.array(index_symbols_quad(*axes))
        assert np.allclose(carlson, quad, rtol=1e-8)


def test_degenerate_oblate_prolate():
    # oblate a = b: A1 == A2 exactly; prolate b = c: A2 == A3 exactly
    A1, A2, A3 = index_symbols(1.0, 1.0, 0.5)
    assert abs(A1 - A2) < 1e-12
    A1, A2, A3 = index_symbols(1.0, 0.5, 0.5)
    assert abs(A2 - A3) < 1e-12


def test_double_symbol_partial_fraction():
    from soloduet.chandrasekhar import _quad_double

    a, b, c = 1.0, 0.7, 0.4
    assert abs(double_symbol(a, b, c, a, b) - _quad_double(a, b, c, a, b)) < 1e-9
    # degenerate route (a ~ b) also agrees with quadrature
    a, b = 1.0, 1.0 - 1e-7
    assert abs(double_symbol(a, b, 0.5, a, b) - _quad_double(a, b, 0.5, a, b)) < 1e-9
