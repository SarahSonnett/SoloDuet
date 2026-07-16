#!/usr/bin/env python
"""Real-object demo: KBO (139775) 2001 QG298 — the classic contact binary.

Fits the actual UH 2.2 m R-band photometry of Sheppard & Jewitt (2004,
AJ 127, 3023; VizieR J/AJ/127/3023) at the published double-peaked period
P = 13.7744 h.  Lacerda & Jewitt (2007) showed this light curve is
well-described by a Roche contact binary with bulk density ~590 kg m^-3;
SoloDuet should reproduce that verdict and density from scratch.  Writes
``docs/images/qg298_fit.png``.

Only the 2003 August-September block is fitted (the 2002 September run is a
different apparition; without distance corrections the two do not share a
mean magnitude).  The phase angle was ~0.6-1.3 deg during the observations
(alpha = 1 deg adopted, as in Lacerda & Jewitt 2007).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soloduet as sd  # noqa: E402
from soloduet.plotting import save_summary  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "2001qg298_sj2004.txt")

PERIOD_HR = 13.7744         # Sheppard & Jewitt (2004), double-peaked
ALPHA_DEG = 1.0             # representative; L&J07 simulate alpha = 1 deg
RHO_PUBLISHED = 590.0       # kg/m3 (Lacerda & Jewitt 2007)


def main() -> int:
    time, mag, merr, _ = sd.read_lightcurve(DATA, object_name="2001 QG298")
    keep = time > 52870.0    # MJD: 2003 Aug-Sep apparition only
    time, mag, merr = time[keep], mag[keep], merr[keep]
    print(f"2001 QG298: {time.size} points from Sheppard & Jewitt (2004), "
          f"2003 apparition, folded at P = {PERIOD_HR} h")

    lc = sd.phase_fold(time, mag, merr, period_hr=PERIOD_HR,
                       alpha_deg=ALPHA_DEG, object_name="2001 QG298")

    print("Fitting both models (a few minutes) ...")
    result = sd.fit_lightcurve(lc, aspect_grid=(90.0, 75.0),
                               ba_step=0.04, q_step=0.05, n_c1=8)
    print()
    print(result.summary())

    outdir = os.path.join(HERE, "docs", "images")
    os.makedirs(outdir, exist_ok=True)
    figpath = os.path.join(outdir, "qg298_fit.png")
    save_summary(result, figpath)
    print(f"\nfigure written to {figpath}")

    # ----- self check vs. the published analysis ---------------------------
    ok = True
    b = result.fits["binary"]
    checks = [
        ("verdict is BINARY (Lacerda & Jewitt 2007)",
         result.verdict.preferred == "binary"),
        (f"density within 25% of the published 590 kg/m3 "
         f"({b.density_kgm3:.0f})",
         abs(b.density_kgm3 - RHO_PUBLISHED) / RHO_PUBLISHED < 0.25),
        (f"components at/near contact (d/(a1+a2) = "
         f"{b.params['separation']:.2f})", b.params["separation"] < 1.25),
        ("amplitude above the 0.9 mag single-figure limit",
         result.verdict.morphology.exceeds_single_limit),
    ]
    for label, passed in checks:
        print(("PASS  " if passed else "FAIL  ") + label)
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
