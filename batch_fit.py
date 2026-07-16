#!/usr/bin/env python
"""Batch SoloDuet runner: fit every asteroid in a survey analysis tree.

Discovers photometry files (default ``Target*.txt``, the SpinDoc/Silhouette
calibrated format with ``Rhelio Delta alpha ... MJD TmagCorr ...`` columns)
under ``--indir``, one asteroid per top-level subdirectory.  Multi-epoch
files of the same object are combined after reducing every point to unit
distances and zero solar phase angle with the IAU H-G phase function
(SpinDoc's ``HGfunction``; Bowell et al. 1989), so campaigns at different
epochs synchronize when folded.  Each object is fit with the full
solo-vs-duet competition and everything is written to one compilation
directory.

Workflow
--------
1. Generate an editable configuration (periods are required; a results
   table like ``TrojanLCs_MyResults.txt`` pre-fills period and G):

       python batch_fit.py --indir .../Analysis --outdir .../SoloDuet_Compilation \\
              --make-config --results-table .../Analysis/TrojanLCs_MyResults.txt

2. Review/edit ``<outdir>/batch_config.csv`` (fill any missing periods,
   delete unwanted objects), then run:

       python batch_fit.py --indir .../Analysis --outdir .../SoloDuet_Compilation [--fast]

Outputs per object: ``<obj>_summary.txt``, ``<obj>_result.json``,
``<obj>_summary.png``.  Survey-wide: ``compilation.csv`` (machine-readable,
appended as each object finishes, so an interrupted run resumes with
``--overwrite`` unset) and ``compilation_summary.txt`` (sorted by dBIC,
duet-like first).

Conventions: periods in HOURS (the full, double-peaked light-curve period);
directories whose names end in ``_dont_include`` are skipped unless
``--include-excluded``; when a file and its ``*_cleaned.txt`` sibling both
exist in the same directory, only the cleaned one is used.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import sys
import time as _time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soloduet as sd  # noqa: E402
from soloduet.cli import _result_json  # noqa: E402
from soloduet.io import read_photometry, reduce_and_correct  # noqa: E402
from soloduet.plotting import save_summary  # noqa: E402

CONFIG_NAME = "batch_config.csv"
DEFAULT_G = 0.15

#: phase drift across the data baseline (from the period uncertainty) above
#: which a combined multi-epoch fold is considered incoherent
SMEAR_WARN = 0.05

CSV_FIELDS = [
    "object", "verdict", "strength", "dbic", "chi2_scale", "n_points",
    "n_files", "span_days", "period_hr", "period_err_hr", "G", "G_err",
    "phase_smear", "alpha_med_deg", "amplitude_mag", "exceeds_0p9_limit",
    "jacobi_redchi2", "jacobi_b_over_a", "jacobi_rho_min_kgm3",
    "free_redchi2", "free_b_over_a", "binary_redchi2", "binary_q",
    "binary_separation", "binary_rho_min_kgm3", "caveats",
]


# ---------------------------------------------------------------------------
# discovery and configuration
# ---------------------------------------------------------------------------

def discover(indir: str, pattern: str, include_excluded: bool) -> dict:
    """Map object name -> list of photometry file paths."""
    groups: dict = {}
    for root, _dirs, files in os.walk(indir):
        rel = os.path.relpath(root, indir)
        if rel == ".":
            continue
        top = rel.split(os.sep)[0]
        if top.endswith("_dont_include") and not include_excluded:
            continue
        obj = top.replace("_dont_include", "")
        matched = [f for f in files if fnmatch.fnmatch(f, pattern)]
        # prefer the *_cleaned sibling when both live in the same directory
        cleaned_stems = {f.replace("_cleaned.txt", "") for f in matched
                         if f.endswith("_cleaned.txt")}
        for f in sorted(matched):
            if (not f.endswith("_cleaned.txt")
                    and f.replace(".txt", "") in cleaned_stems):
                continue
            groups.setdefault(obj, []).append(os.path.join(root, f))
    return dict(sorted(groups.items()))


def read_results_table(path: str) -> dict:
    """Parse a vetted results table keyed on its first column (object name).

    Expects columns ``Per``/``PerErr`` (hours) and ``G``/``GErr``; negative
    values mean 'no viable solution' (both parameters), and such objects are
    left blank in the config so they are skipped until a period is supplied.
    """
    out = {}
    with open(path) as fh:
        header = fh.readline().split()
        cols = {name.lower(): i for i, name in enumerate(header)}
        for line in fh:
            parts = line.split()
            if len(parts) < len(header) - 1 or not parts[0].strip():
                continue
            name = parts[0]

            def get(col):
                try:
                    return float(parts[cols[col]])
                except (KeyError, ValueError, IndexError):
                    return None

            per, per_err = get("per"), get("pererr")
            g, g_err = get("g"), get("gerr")
            out[name] = {
                "period_hr": per if (per is not None and per > 0) else None,
                "period_err_hr": per_err
                if (per_err is not None and per_err >= 0) else None,
                # G may be legitimately (slightly) negative; the -10
                # sentinel marks 'no viable solution'
                "G": g if (g is not None and -1.0 < g < 1.0) else None,
                "G_err": g_err
                if (g_err is not None and 0 <= g_err < 1.0) else None,
            }
    return out


def make_config(groups: dict, outpath: str, results: dict) -> None:
    with open(outpath, "w", newline="") as fh:
        w = csv.writer(fh)
        fh.write("# SoloDuet batch configuration -- review before running.\n")
        fh.write("# period_hr is the FULL (double-peaked) light-curve period"
                 " in hours; it is required (rows without one are skipped).\n")
        fh.write(f"# G is the IAU H-G slope parameter (blank -> {DEFAULT_G}).\n")
        fh.write("# period_err_hr is used to warn when a multi-epoch fold"
                 " loses phase coherence.\n")
        w.writerow(["object", "period_hr", "period_err_hr", "G", "G_err",
                    "files"])
        for obj, files in groups.items():
            info = results.get(obj, {})
            per = info.get("period_hr")
            per_err = info.get("period_err_hr")
            g = info.get("G")
            g_err = info.get("G_err")
            w.writerow([obj,
                        f"{per:.6g}" if per else "",
                        f"{per_err:.6g}" if (per and per_err is not None) else "",
                        f"{g:.2f}" if g is not None else "",
                        f"{g_err:.2f}" if g_err is not None else "",
                        ";".join(files)])
    n_missing = sum(1 for o in groups if not results.get(o, {}).get("period_hr"))
    print(f"wrote {outpath}: {len(groups)} objects "
          f"({n_missing} still need a period_hr)")


def read_config(path: str) -> list:
    rows = []
    with open(path) as fh:
        rdr = csv.reader(line for line in fh if not line.startswith("#"))
        header = next(rdr)
        idx = {name: i for i, name in enumerate(header)}
        def opt(parts, col, default=None):
            if col not in idx or not parts[idx[col]].strip():
                return default
            return float(parts[idx[col]])

        for parts in rdr:
            if not parts or not parts[0].strip():
                continue
            rows.append({
                "object": parts[idx["object"]].strip(),
                "period_hr": opt(parts, "period_hr"),
                "period_err_hr": opt(parts, "period_err_hr"),
                "G": opt(parts, "G", DEFAULT_G),
                "G_err": opt(parts, "G_err"),
                "files": [f for f in parts[idx["files"]].split(";") if f],
            })
    return rows


# ---------------------------------------------------------------------------
# per-object pipeline
# ---------------------------------------------------------------------------

def load_object(entry: dict):
    """Read, reduce, H-G-correct, and merge all epochs of one object."""
    times, mags, errs, alphas = [], [], [], []
    for path in entry["files"]:
        phot = read_photometry(path, object_name=entry["object"])
        times.append(phot.time)
        mags.append(reduce_and_correct(phot, entry["G"]))
        errs.append(phot.merr)
        alphas.append(phot.alpha)
    t = np.concatenate(times)
    m = np.concatenate(mags)
    e = np.concatenate(errs)
    a = np.concatenate(alphas)
    good = np.isfinite(t) & np.isfinite(m) & np.isfinite(e)
    t, m, e, a = t[good], m[good], e[good], a[good]
    lc = sd.phase_fold(t, m, e, period_hr=entry["period_hr"],
                       alpha_deg=float(np.median(a)),
                       object_name=entry["object"])
    span = float(t.max() - t.min())
    return lc, span


def fit_object(entry: dict, outdir: str, fit_kw: dict) -> dict:
    lc, span = load_object(entry)

    # phase-coherence check: drift of the fold across the data baseline
    # implied by the period uncertainty
    smear = ""
    smear_note = None
    if entry.get("period_err_hr"):
        p_days = entry["period_hr"] / 24.0
        smear = round((span / p_days)
                      * (entry["period_err_hr"] / entry["period_hr"]), 3)
        if smear > SMEAR_WARN:
            smear_note = (f"period uncertainty smears the fold by "
                          f"{smear:.2f} in phase across the {span:.0f}-day "
                          "baseline: the combined-epoch fold is not "
                          "coherent; fit campaigns separately or refine "
                          "the period first")
            print(f"    WARNING: {smear_note}")

    result = sd.fit_lightcurve(lc, **fit_kw)
    if smear_note:
        result.verdict.caveats.insert(0, smear_note)

    stem = entry["object"].replace(" ", "_")
    with open(os.path.join(outdir, f"{stem}_summary.txt"), "w") as fh:
        fh.write(result.summary() + "\n")
    with open(os.path.join(outdir, f"{stem}_result.json"), "w") as fh:
        json.dump(_result_json(result), fh, indent=2)
    save_summary(result, os.path.join(outdir, f"{stem}_summary.png"))

    j = result.fits.get("jacobi")
    f = result.fits.get("free")
    b = result.fits.get("binary")
    v = result.verdict
    return {
        "object": entry["object"],
        "verdict": v.preferred,
        "strength": v.strength,
        "dbic": round(v.delta_bic, 1),
        "chi2_scale": round(v.chi2_scale, 2),
        "n_points": len(lc),
        "n_files": len(entry["files"]),
        "span_days": round(span, 2),
        "period_hr": entry["period_hr"],
        "period_err_hr": entry.get("period_err_hr") or "",
        "G": entry["G"],
        "G_err": entry.get("G_err") or "",
        "phase_smear": smear,
        "alpha_med_deg": round(lc.alpha_deg, 2),
        "amplitude_mag": round(v.morphology.amplitude, 3),
        "exceeds_0p9_limit": v.morphology.exceeds_single_limit,
        "jacobi_redchi2": round(j.redchi2, 2) if j else "",
        "jacobi_b_over_a": round(j.params["b_over_a"], 3) if j else "",
        "jacobi_rho_min_kgm3": round(j.density_kgm3) if j else "",
        "free_redchi2": round(f.redchi2, 2) if f else "",
        "free_b_over_a": round(f.params["b_over_a"], 3) if f else "",
        "binary_redchi2": round(b.redchi2, 2) if b else "",
        "binary_q": round(b.params["q"], 2) if b else "",
        "binary_separation": round(b.params["separation"], 3) if b else "",
        "binary_rho_min_kgm3": round(b.density_kgm3) if b else "",
        "caveats": " | ".join(v.caveats),
    }


def write_summary_table(rows: list, path: str) -> None:
    order = np.argsort([-r["dbic"] for r in rows])
    with open(path, "w") as fh:
        fh.write("SoloDuet batch compilation -- sorted by dBIC "
                 "(duet-like first; dBIC > 0 favors the binary)\n")
        fh.write("=" * 100 + "\n")
        fh.write("(binary rho values are MINIMA: the aspect angle is "
                 "unconstrained)\n")
        fh.write("%-10s %-14s %-13s %8s %6s %6s %8s %8s %8s %11s %6s\n" % (
            "object", "verdict", "strength", "dBIC", "n", "amp",
            "chi2_jac", "chi2_bin", "q", "rho_bin_min", "smear"))
        for i in order:
            r = rows[i]
            fh.write("%-10s %-14s %-13s %8.1f %6d %6.2f %8s %8s %8s %11s %6s\n"
                     % (r["object"], r["verdict"], r["strength"], r["dbic"],
                        r["n_points"], r["amplitude_mag"],
                        r["jacobi_redchi2"], r["binary_redchi2"],
                        r["binary_q"], r["binary_rho_min_kgm3"],
                        r.get("phase_smear", "")))
        flagged = [rows[i]["object"] for i in order
                   if rows[i]["exceeds_0p9_limit"]]
        if flagged:
            fh.write("\nranges above the 0.9 mag strengthless single-figure "
                     "limit: " + ", ".join(flagged) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="batch_fit",
        description="Run SoloDuet on every asteroid in a survey tree.")
    p.add_argument("--indir", required=True,
                   help="analysis tree (one asteroid per subdirectory)")
    p.add_argument("--outdir", required=True, help="compilation directory")
    p.add_argument("--pattern", default="Target*.txt",
                   help="photometry filename pattern (default Target*.txt)")
    p.add_argument("--config", default=None,
                   help=f"configuration CSV (default <outdir>/{CONFIG_NAME})")
    p.add_argument("--make-config", action="store_true",
                   help="discover files, write the config template, and exit")
    p.add_argument("--results-table", default=None,
                   help="optional results table (Name/G/Per columns) used to "
                        "pre-fill the config in --make-config mode")
    p.add_argument("--objects", nargs="+", default=None,
                   help="fit only these objects")
    p.add_argument("--include-excluded", action="store_true",
                   help="also process *_dont_include directories")
    p.add_argument("--free-ellipsoid", action="store_true",
                   help="also fit the unconstrained triaxial mode")
    p.add_argument("--fast", action="store_true",
                   help="coarser grids/renders (~4x faster per object)")
    p.add_argument("--overwrite", action="store_true",
                   help="refit objects whose outputs already exist")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    config_path = args.config or os.path.join(args.outdir, CONFIG_NAME)

    if args.make_config:
        groups = discover(args.indir, args.pattern, args.include_excluded)
        results = (read_results_table(args.results_table)
                   if args.results_table else {})
        make_config(groups, config_path, results)
        return 0

    if not os.path.exists(config_path):
        print(f"no config at {config_path}; run with --make-config first")
        return 1

    entries = read_config(config_path)
    if args.objects:
        entries = [e for e in entries if e["object"] in set(args.objects)]
    runnable = [e for e in entries if e["period_hr"]]
    skipped = [e["object"] for e in entries if not e["period_hr"]]
    if skipped:
        print("skipping (no period in config): " + ", ".join(skipped))

    fit_kw = dict(free_ellipsoid=args.free_ellipsoid)
    if args.fast:
        fit_kw.update(ba_step=0.05, q_step=0.1, n_c1=5,
                      grid_n_pixels=64, grid_n_phases=48,
                      n_pixels=128, n_phases=96)

    csv_path = os.path.join(args.outdir, "compilation.csv")
    done = set()
    if os.path.exists(csv_path) and not args.overwrite:
        with open(csv_path) as fh:
            done = {r["object"] for r in csv.DictReader(fh)}

    rows = []
    if done:
        with open(csv_path) as fh:
            for r in csv.DictReader(fh):
                r["dbic"] = float(r["dbic"])
                r["n_points"] = int(r["n_points"])
                r["amplitude_mag"] = float(r["amplitude_mag"])
                r["exceeds_0p9_limit"] = r["exceeds_0p9_limit"] == "True"
                rows.append(r)

    new_file = not os.path.exists(csv_path) or args.overwrite
    mode = "w" if new_file else "a"
    with open(csv_path, mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for k, entry in enumerate(runnable):
            if entry["object"] in done and not args.overwrite:
                print(f"[{k + 1}/{len(runnable)}] {entry['object']}: "
                      "already in compilation, skipping")
                continue
            t0 = _time.time()
            print(f"[{k + 1}/{len(runnable)}] {entry['object']} "
                  f"(P = {entry['period_hr']} h, {len(entry['files'])} "
                  f"file(s)) ...", flush=True)
            try:
                row = fit_object(entry, args.outdir, fit_kw)
            except Exception as exc:  # keep the batch alive
                print(f"    FAILED: {exc}")
                continue
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            print(f"    {row['verdict'].upper()} ({row['strength']}, "
                  f"dBIC = {row['dbic']:+.1f})  [{_time.time() - t0:.0f} s]")

    if rows:
        write_summary_table(rows, os.path.join(args.outdir,
                                               "compilation_summary.txt"))
        print(f"\ncompilation written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
