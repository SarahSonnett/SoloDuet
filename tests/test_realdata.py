"""End-to-end regression on the real 2001 QG298 photometry (slow).

Reproduces the Lacerda & Jewitt (2007) result on the Sheppard & Jewitt
(2004) data shipped in data/2001qg298_sj2004.txt.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soloduet as sd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "2001qg298_sj2004.txt")


def test_data_file_loads():
    time, mag, merr, _ = sd.read_lightcurve(DATA, object_name="2001 QG298")
    assert time.size == 113
    assert 21.2 < np.min(mag) < 21.4
    assert 22.4 < np.max(mag) < 22.6


@pytest.mark.slow
def test_qg298_contact_binary_verdict():
    time, mag, merr, _ = sd.read_lightcurve(DATA, object_name="2001 QG298")
    keep = time > 52870.0  # 2003 apparition
    lc = sd.phase_fold(time[keep], mag[keep], merr[keep],
                       period_hr=13.7744, alpha_deg=1.0,
                       object_name="2001 QG298")
    res = sd.fit_lightcurve(lc, aspect_grid=(90.0, 75.0), ba_step=0.05,
                            q_step=0.1, n_c1=6)
    assert res.verdict.preferred == "binary"
    b = res.fits["binary"]
    # Lacerda & Jewitt (2007): rho = 590 (+143/-47) lunar, ~660 icy;
    # d/(A+a) = 0.90 (+0.31/-0.14)
    assert 450.0 < b.density_kgm3 < 850.0
    assert b.params["separation"] < 1.3
    assert res.verdict.morphology.exceeds_single_limit
