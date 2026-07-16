#!/usr/bin/env python
"""Self-checking DUET round trip.

Synthesizes noisy photometry of a known near-contact Roche binary, runs the
full solo-vs-duet competition, and verifies that the binary model wins with
the injected mass ratio and density recovered.  Writes
``docs/images/duet_roundtrip.png`` (the README hero figure).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soloduet as sd  # noqa: E402
from soloduet.compare import interp_periodic  # noqa: E402
from soloduet.jacobi import density_from_spin  # noqa: E402
from soloduet.models import binary_scene, duet_curve  # noqa: E402
from soloduet.plotting import save_summary  # noqa: E402

TRUE_Q = 0.80
TRUE_C1 = 0.60
TRUE_ASPECT = 90.0
TRUE_PHI0 = 0.13
PERIOD_HR = 13.7744
ALPHA_DEG = 1.0
NOISE_MAG = 0.03
N_POINTS = 90


def main() -> int:
    rng = np.random.default_rng(298)
    _, _, _, w2, sep = binary_scene(TRUE_Q, TRUE_C1)
    rho_true = density_from_spin(w2, PERIOD_HR)
    print(f"Synthesizing a Roche duet: q = {TRUE_Q}, c1 = {TRUE_C1} "
          f"(separation {sep:.2f}, rho = {rho_true:.0f} kg/m3), "
          f"P = {PERIOD_HR} h ...")
    mph, mdm = duet_curve(TRUE_Q, TRUE_C1, TRUE_ASPECT, ALPHA_DEG, "lunar",
                          n_phases=128, n_pixels=192)
    phase = np.sort(rng.uniform(0, 1, N_POINTS))
    dmag = interp_periodic(phase - TRUE_PHI0, mph, mdm) \
        + rng.normal(0, NOISE_MAG, N_POINTS)
    lc = sd.FoldedLightcurve(phase=phase, dmag=dmag,
                             err=np.full(N_POINTS, NOISE_MAG),
                             period_hr=PERIOD_HR, alpha_deg=ALPHA_DEG,
                             object_name="synthetic duet")

    print("Fitting both models (a couple of minutes) ...")
    result = sd.fit_lightcurve(lc, aspect_grid=(90.0, 75.0, 60.0),
                               ba_step=0.05, q_step=0.1, n_c1=6)
    print()
    print(result.summary())

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "docs", "images")
    os.makedirs(outdir, exist_ok=True)
    figpath = os.path.join(outdir, "duet_roundtrip.png")
    save_summary(result, figpath)
    print(f"\nfigure written to {figpath}")

    # ----- self check ------------------------------------------------------
    ok = True
    b = result.fits["binary"]
    checks = [
        ("verdict is BINARY", result.verdict.preferred == "binary"),
        (f"recovered q within 0.15 ({b.params['q']:.2f} vs {TRUE_Q})",
         abs(b.params["q"] - TRUE_Q) < 0.15),
        (f"density within 15% ({b.density_kgm3:.0f} vs {rho_true:.0f})",
         abs(b.density_kgm3 - rho_true) / rho_true < 0.15),
        (f"reduced chi2 sensible ({b.redchi2:.2f})", b.redchi2 < 2.0),
        ("amplitude flagged above the 0.9 mag single-figure limit",
         result.verdict.morphology.exceeds_single_limit),
    ]
    for label, passed in checks:
        print(("PASS  " if passed else "FAIL  ") + label)
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
