#!/usr/bin/env python
"""Field test on two ground-truth objects: a contact binary and a single.

Runs the archived DAMIT photometry of

* **(216) Kleopatra** — radar/AO-verified bilobed ("dog-bone") body
  (Ostro et al. 2000; Shepard et al. 2018), bulk density 3.38 g/cm3 from
  the orbits of its two moons (Marchis et al. 2021), and
* **(433) Eros** — the spacecraft-verified elongated *single* asteroid
  (NEAR Shoemaker: 34.4 x 11.2 x 11.2 km, rho = 2.67 g/cm3; Veverka et
  al. 2000),

through the identical SoloDuet pipeline, and checks that the verdicts land
on the right sides.  Both light curves have peak-to-peak ranges just above
the 0.9 mag strengthless single-figure limit — Kleopatra because it *is*
two lobes, Eros because it is a strength-dominated monolith — so this pair
also exercises the honest-caveat machinery.  Writes
``docs/images/kleopatra_fit.png`` and ``docs/images/eros_fit.png``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soloduet as sd  # noqa: E402
from soloduet.plotting import save_summary  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

TARGETS = [
    dict(infile="data/216kleopatra_damit.txt", name="(216) Kleopatra",
         period_hr=5.385282, alpha_deg=6.0, aspect_deg=83.6,
         figure="kleopatra_fit.png"),
    dict(infile="data/433eros_damit.txt", name="(433) Eros",
         period_hr=5.27025528, alpha_deg=8.8, aspect_deg=84.3,
         figure="eros_fit.png"),
]


def fit_target(t):
    time, mag, merr, _ = sd.read_lightcurve(os.path.join(HERE, t["infile"]),
                                            object_name=t["name"])
    lc = sd.phase_fold(time, mag, merr, period_hr=t["period_hr"],
                       alpha_deg=t["alpha_deg"], object_name=t["name"])
    print(f"\n=== {t['name']}: {len(lc)} points, "
          f"amplitude {lc.amplitude:.2f} mag, "
          f"aspect fixed at {t['aspect_deg']} deg (known pole)")
    result = sd.fit_lightcurve(lc, free_ellipsoid=True,
                               aspect_grid=(t["aspect_deg"],),
                               ba_step=0.03, q_step=0.05, n_c1=8)
    print(result.summary())
    outdir = os.path.join(HERE, "docs", "images")
    os.makedirs(outdir, exist_ok=True)
    save_summary(result, os.path.join(outdir, t["figure"]))
    print(f"figure written to docs/images/{t['figure']}")
    return result


def main() -> int:
    kleo = fit_target(TARGETS[0])
    eros = fit_target(TARGETS[1])

    ok = True
    kb = kleo.fits["binary"]
    checks = [
        ("Kleopatra verdict is BINARY (radar ground truth: bilobed)",
         kleo.verdict.preferred == "binary"),
        (f"Kleopatra density within 35% of the 3.38 g/cm3 satellite-orbit "
         f"value ({kb.density_kgm3 / 1000:.2f} g/cm3)",
         abs(kb.density_kgm3 - 3380.0) / 3380.0 < 0.35),
        (f"Kleopatra components at/near contact "
         f"(d/(a1+a2) = {kb.params['separation']:.2f})",
         kb.params["separation"] < 1.15),
        ("Eros verdict is SINGLE (spacecraft ground truth: one body)",
         eros.verdict.preferred == "single"),
        ("Eros: binary model decisively worse than the single models",
         eros.fits["binary"].redchi2
         > 1.3 * min(eros.fits["jacobi"].redchi2,
                     eros.fits["free"].redchi2)),
        (f"Eros free-ellipsoid b/a = "
         f"{eros.fits['free'].params['b_over_a']:.2f} near the NEAR value "
         "0.33 (convex-model curves are typically slightly rounder)",
         0.28 <= eros.fits["free"].params["b_over_a"] <= 0.48),
        ("both curves flagged above the 0.9 mag strengthless limit",
         kleo.verdict.morphology.exceeds_single_limit
         and eros.verdict.morphology.exceeds_single_limit),
    ]
    print()
    for label, passed in checks:
        print(("PASS  " if passed else "FAIL  ") + label)
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
