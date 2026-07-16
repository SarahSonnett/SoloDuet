"""Command-line interface: ``soloduet`` (or ``python fit_duet.py``)."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import __version__
from .fit import DEFAULT_ASPECT_GRID, fit_lightcurve
from .io import phase_fold, read_lightcurve
from .plotting import save_summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="soloduet",
        description="Fit a single Jacobi ellipsoid (solo) and a Roche "
                    "contact/close binary (duet) to a rotational light "
                    "curve and report which model the data prefer.")
    p.add_argument("--infile", required=True,
                   help="photometry table: either a bare 'time mag [merr]' "
                        "file (JD or MJD) or a headered table with "
                        "time/mag/merr/rhelio/delta/alpha columns")
    p.add_argument("--period-hr", type=float, required=True,
                   help="light-curve period in HOURS (one rotation of a "
                        "single body = one orbit of a tidally locked binary; "
                        "use the double-peaked period)")
    p.add_argument("--object", default=None, help="target designation")
    p.add_argument("--alpha", type=float, default=None,
                   help="solar phase angle [deg] (overrides the value "
                        "derived from the input table; default 0 for bare "
                        "tables)")
    p.add_argument("--aspect", type=float, nargs="+", default=None,
                   help="aspect angle(s) [deg] to try (default: grid "
                        f"{DEFAULT_ASPECT_GRID}); pass one value if the "
                        "pole is known")
    p.add_argument("--scattering", nargs="+", default=["lunar", "icy"],
                   choices=["lunar", "icy", "geometric"],
                   help="scattering law(s) to try (default: lunar icy). "
                        "'geometric' = brightness proportional to the "
                        "illuminated cross-section (Surdej & Surdej 1978 "
                        "analytic curve for the solo models)")
    p.add_argument("--free-ellipsoid", action="store_true",
                   help="also fit an unconstrained triaxial ellipsoid "
                        "(no density inference)")
    p.add_argument("--jd-min", type=float, default=None,
                   help="only use points with JD/MJD >= this value")
    p.add_argument("--jd-max", type=float, default=None,
                   help="only use points with JD/MJD <= this value")
    p.add_argument("--fast", action="store_true",
                   help="coarser grids and renders (quick look)")
    p.add_argument("--n-pixels", type=int, default=None,
                   help="render resolution for the final models "
                        "(default 192, or 128 with --fast)")
    p.add_argument("--outdir", default="results", help="output directory")
    p.add_argument("--version", action="version", version=__version__)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    time, mag, merr, alpha = read_lightcurve(args.infile,
                                             object_name=args.object)
    if args.jd_min is not None or args.jd_max is not None:
        lo = -np.inf if args.jd_min is None else args.jd_min - 2_400_000.5 \
            if args.jd_min > 2_400_000 else args.jd_min
        hi = np.inf if args.jd_max is None else args.jd_max - 2_400_000.5 \
            if args.jd_max > 2_400_000 else args.jd_max
        keep = (time >= lo) & (time <= hi)
        time, mag, merr = time[keep], mag[keep], merr[keep]
    if args.alpha is not None:
        alpha = args.alpha

    lc = phase_fold(time, mag, merr, period_hr=args.period_hr,
                    alpha_deg=alpha, object_name=args.object)

    kw = {}
    if args.fast:
        kw.update(ba_step=0.05, q_step=0.1, n_c1=5,
                  grid_n_pixels=64, grid_n_phases=48,
                  n_pixels=128, n_phases=96)
    if args.aspect:
        kw["aspect_grid"] = tuple(args.aspect)
    if args.n_pixels is not None:
        kw["n_pixels"] = args.n_pixels

    result = fit_lightcurve(lc, free_ellipsoid=args.free_ellipsoid,
                            scatterings=tuple(args.scattering), **kw)

    print(result.summary())

    os.makedirs(args.outdir, exist_ok=True)
    stem = (args.object or os.path.splitext(
        os.path.basename(args.infile))[0]).replace(" ", "_")
    txt_path = os.path.join(args.outdir, f"{stem}_summary.txt")
    png_path = os.path.join(args.outdir, f"{stem}_summary.png")
    json_path = os.path.join(args.outdir, f"{stem}_result.json")

    with open(txt_path, "w") as fh:
        fh.write(result.summary() + "\n")
    save_summary(result, png_path)
    with open(json_path, "w") as fh:
        json.dump(_result_json(result), fh, indent=2)
    print(f"\nwrote {txt_path}, {png_path}, {json_path}")
    return 0


def _result_json(result) -> dict:
    out = {
        "object": result.lc.object_name,
        "period_hr": result.period_hr,
        "n_points": len(result.lc),
        "alpha_deg": result.lc.alpha_deg,
        "verdict": {
            "preferred": result.verdict.preferred,
            "strength": result.verdict.strength,
            "delta_bic": result.verdict.delta_bic,
            "delta_aic": result.verdict.delta_aic,
            "f_test_p": result.verdict.f_test_p,
            "caveats": result.verdict.caveats,
        },
        "morphology": {
            "amplitude_mag": result.verdict.morphology.amplitude,
            "exceeds_single_limit":
                bool(result.verdict.morphology.exceeds_single_limit),
            "minima_curvature_ratio":
                result.verdict.morphology.minima_curvature_ratio,
            "odd_harmonic_fraction":
                result.verdict.morphology.odd_harmonic_fraction,
        },
        "models": {},
    }
    for name, fit in result.fits.items():
        out["models"][name] = {
            "scattering": fit.scattering,
            "params": {k: float(v) for k, v in fit.params.items()},
            "chi2": fit.chi2,
            "dof": fit.dof,
            "reduced_chi2": fit.redchi2,
            "bic": fit.bic,
            "aic": fit.aic,
            "density_kgm3": fit.density_kgm3,
            "within_1sigma": {k: list(v) for k, v in fit.within_1sig.items()},
            "notes": fit.notes,
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
