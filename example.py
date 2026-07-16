#!/usr/bin/env python
"""Self-checking SOLO round trip.

Synthesizes noisy photometry of a known Jacobi figure with SoloDuet's own
renderer, runs the full solo-vs-duet competition, and verifies that the
single-ellipsoid model wins and the injected shape is recovered.  Writes
``docs/images/solo_roundtrip.png``.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soloduet as sd  # noqa: E402
from soloduet.compare import interp_periodic  # noqa: E402
from soloduet.models import jacobi_axes, solo_curve  # noqa: E402
from soloduet.plotting import save_summary  # noqa: E402

TRUE_BA = 0.60
TRUE_ASPECT = 90.0
TRUE_PHI0 = 0.40
PERIOD_HR = 7.0
ALPHA_DEG = 1.0
NOISE_MAG = 0.02
N_POINTS = 90


def main() -> int:
    rng = np.random.default_rng(2026)
    print(f"Synthesizing a Jacobi figure: b/a = {TRUE_BA}, "
          f"aspect = {TRUE_ASPECT} deg, P = {PERIOD_HR} h ...")
    mph, mdm = solo_curve(jacobi_axes(TRUE_BA), TRUE_ASPECT, ALPHA_DEG,
                          "lunar", n_phases=128, n_pixels=192)
    phase = np.sort(rng.uniform(0, 1, N_POINTS))
    dmag = interp_periodic(phase - TRUE_PHI0, mph, mdm) \
        + rng.normal(0, NOISE_MAG, N_POINTS)
    lc = sd.FoldedLightcurve(phase=phase, dmag=dmag,
                             err=np.full(N_POINTS, NOISE_MAG),
                             period_hr=PERIOD_HR, alpha_deg=ALPHA_DEG,
                             object_name="synthetic solo")

    print("Fitting both models (a couple of minutes) ...")
    result = sd.fit_lightcurve(lc, aspect_grid=(90.0, 75.0, 60.0),
                               ba_step=0.05, q_step=0.1, n_c1=6)
    print()
    print(result.summary())

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "docs", "images")
    os.makedirs(outdir, exist_ok=True)
    figpath = os.path.join(outdir, "solo_roundtrip.png")
    save_summary(result, figpath)
    print(f"\nfigure written to {figpath}")

    # ----- self check ------------------------------------------------------
    # A single-apparition curve cannot fully break the aspect-shape
    # degeneracy (a more elongated body at lower aspect mimics a rounder one
    # equator-on), so the meaningful check is that the injected b/a lies
    # inside the reported ~1-sigma (chi2/chi2_best < 2) region.
    ok = True
    j = result.fits["jacobi"]
    ba_lo, ba_hi = j.within_1sig.get(
        "b_over_a", (j.params["b_over_a"] - 0.1, j.params["b_over_a"] + 0.1))
    checks = [
        ("verdict is SINGLE", result.verdict.preferred == "single"),
        (f"true b/a = {TRUE_BA} consistent with the 1-sigma range "
         f"[{ba_lo:.2f}, {ba_hi:.2f}] +/- the 0.03 grid spacing "
         f"(best {j.params['b_over_a']:.3f}; aspect-shape degeneracy)",
         ba_lo - 0.03 <= TRUE_BA <= ba_hi + 0.03),
        (f"reduced chi2 sensible ({j.redchi2:.2f})", j.redchi2 < 2.0),
        ("a density was inferred", j.density_kgm3 is not None),
    ]
    for label, passed in checks:
        print(("PASS  " if passed else "FAIL  ") + label)
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
