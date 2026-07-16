"""Fit orchestrator: solo vs. duet model competition on a folded light curve.

For each requested model class and each scattering law the fitter runs a
grid stage over the precomputed shape sequences x aspect angles (rotational
phase offset scanned, zero-point analytic — both free of rendering cost),
then refines the best node with Nelder-Mead over the continuous parameters.
Model classes are compared with Delta-BIC (Kass & Raftery 1995) and the
Lacerda & Jewitt (2007) chi^2/chi^2_best < 2 criterion provides approximate
1-sigma parameter ranges; morphology diagnostics corroborate the verdict.

Free-parameter counts: Jacobi solo k = 4 (b/a, aspect, phi0, zero-point);
free ellipsoid k = 5 (+ c/b); Roche binary k = 5 (q, c1, aspect, phi0,
zero-point).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from . import compare, models
from .io import FoldedLightcurve
from .jacobi import BA_MIN, density_from_spin

DEFAULT_ASPECT_GRID = (90.0, 80.0, 70.0, 60.0, 50.0, 40.0)
SCATTERING_LAWS = ("lunar", "icy")


@dataclass
class GridRecord:
    """One evaluated grid node (kept for confidence regions and plots)."""

    params: Dict[str, float]
    chi2: float


@dataclass
class ModelFit:
    """Best fit of one model class (best over the scattering laws tried)."""

    name: str                       # "jacobi" | "free" | "binary"
    scattering: str
    params: Dict[str, float]
    chi2: float
    n: int
    k: int
    curve: Tuple[np.ndarray, np.ndarray]      # dense (phase, dmag), unshifted
    density_kgm3: Optional[float] = None
    within_1sig: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    grid: List[GridRecord] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def dof(self) -> int:
        return max(self.n - self.k, 1)

    @property
    def redchi2(self) -> float:
        return self.chi2 / self.dof

    @property
    def aic(self) -> float:
        return compare.aic(self.chi2, self.k, self.n)

    @property
    def bic(self) -> float:
        return compare.bic(self.chi2, self.k, self.n)


@dataclass
class Verdict:
    """The solo-vs-duet decision with its statistical basis."""

    preferred: str                 # "single" | "binary" | "indeterminate"
    strength: str                  # Kass & Raftery ladder
    delta_bic: float               # BIC(single) - BIC(binary); >0 favors binary
    delta_aic: float
    f_test_p: float
    morphology: compare.MorphologyReport
    caveats: List[str] = field(default_factory=list)


@dataclass
class SoloDuetResult:
    """Everything :func:`fit_lightcurve` learned about one light curve."""

    lc: FoldedLightcurve
    fits: Dict[str, ModelFit]
    verdict: Verdict
    period_hr: float

    def summary(self) -> str:
        lines = ["SoloDuet fit summary", "=" * 60]
        if self.lc.object_name:
            lines.append(f"object        : {self.lc.object_name}")
        lines += [
            f"data points   : {len(self.lc)}",
            f"period        : {self.period_hr:.4f} h "
            f"(one rotation / one binary orbit)",
            f"phase angle   : {self.lc.alpha_deg:.2f} deg",
            "",
        ]
        for name in ("jacobi", "free", "binary"):
            fit = self.fits.get(name)
            if fit is None:
                continue
            lines.append(f"--- {name} model ({fit.scattering} scattering) ---")
            for key, val in fit.params.items():
                rng = fit.within_1sig.get(key)
                span = (f"   [{rng[0]:.3f}, {rng[1]:.3f}]" if rng else "")
                lines.append(f"  {key:<12}: {val:.4f}{span}")
            lines.append(f"  chi2/dof    : {fit.chi2:.1f}/{fit.dof} "
                         f"= {fit.redchi2:.2f}   BIC = {fit.bic:.1f}")
            if fit.density_kgm3 is not None:
                lines.append(f"  bulk density: {fit.density_kgm3:.0f} kg/m3 "
                             f"({fit.density_kgm3 / 1000.0:.2f} g/cm3)")
            for note in fit.notes:
                lines.append(f"  note        : {note}")
            lines.append("")
        v = self.verdict
        lines += ["--- verdict ---",
                  f"  preferred   : {v.preferred.upper()} "
                  f"({v.strength}, dBIC = {v.delta_bic:+.1f})",
                  f"  dAIC = {v.delta_aic:+.1f}   F-test p = {v.f_test_p:.3f} "
                  f"(heuristic)"]
        lines += ["  morphology  : " + s for s in v.morphology.summary_lines()]
        for cav in v.caveats:
            lines.append(f"  caveat      : {cav}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# per-model machinery
# ---------------------------------------------------------------------------

def _eval_node(lc, curve_fn, n_phi0: int = 64, polish: bool = False):
    """(chi2, phi0, zp) of one rendered node against the data."""
    ph, dm = curve_fn()
    return compare.scan_phi0(lc, ph, dm, n_scan=n_phi0, polish=polish)


def _jacobi_nodes(ba_step: float) -> np.ndarray:
    return np.arange(BA_MIN, 1.0 + 1e-9, ba_step)


def _binary_nodes(q_step: float, n_c1: int) -> List[Tuple[float, float]]:
    fams = models.binary_families()
    q_all = np.array(sorted(fams))
    ratio = q_all / q_step
    q_sel = q_all[np.isclose(ratio, np.round(ratio), atol=1e-6)]
    if q_sel.size == 0:
        q_sel = q_all
    nodes = []
    for q in q_sel:
        c1s = fams[float(q)]["c1"]
        take = np.unique(np.linspace(0, c1s.size - 1,
                                     min(n_c1, c1s.size)).astype(int))
        nodes.extend((float(q), float(c1s[i])) for i in take)
    return nodes


def _within_ranges(grid: List[GridRecord], chi2_best: float) -> Dict[str, Tuple[float, float]]:
    """Per-parameter ranges of nodes with chi2/chi2_best < 2 (L&J07)."""
    good = [g for g in grid if g.chi2 < 2.0 * chi2_best]
    if not good:
        return {}
    out: Dict[str, Tuple[float, float]] = {}
    for key in good[0].params:
        vals = [g.params[key] for g in good]
        out[key] = (float(min(vals)), float(max(vals)))
    return out


def _grid_stage(name, lc, law, aspect_grid, render_kw):
    """Coarse-resolution grid scan; returns (records, best_chi2, best_params)."""
    gp = dict(n_phases=render_kw["grid_n_phases"],
              n_pixels=render_kw["grid_n_pixels"])
    grid: List[GridRecord] = []
    best = None

    def consider(curve_fn, params):
        chi2, phi0, zp = _eval_node(lc, curve_fn)
        params = dict(params, phi0=phi0, zeropoint=zp)
        grid.append(GridRecord(params, chi2))
        return chi2, params

    if name == "jacobi":
        for ba in _jacobi_nodes(render_kw["ba_step"]):
            axes = models.jacobi_axes(ba)
            for theta in aspect_grid:
                chi2, params = consider(
                    lambda: models.solo_curve(axes, theta, lc.alpha_deg,
                                              law, **gp),
                    {"b_over_a": float(ba), "aspect_deg": float(theta)})
                if best is None or chi2 < best[0]:
                    best = (chi2, params)
    elif name == "free":
        for ba in np.arange(0.3, 1.0 + 1e-9, 0.05):
            for cb in np.arange(0.4, 1.0 + 1e-9, 0.1):
                axes = (1.0, float(ba), float(ba * cb))
                for theta in aspect_grid:
                    chi2, params = consider(
                        lambda: models.solo_curve(axes, theta, lc.alpha_deg,
                                                  law, **gp),
                        {"b_over_a": float(ba), "c_over_b": float(cb),
                         "aspect_deg": float(theta)})
                    if best is None or chi2 < best[0]:
                        best = (chi2, params)
    elif name == "binary":
        for q, c1 in _binary_nodes(render_kw["q_step"], render_kw["n_c1"]):
            for theta in aspect_grid:
                chi2, params = consider(
                    lambda: models.duet_curve(q, c1, theta, lc.alpha_deg,
                                              law, **gp),
                    {"q": q, "c1": c1, "aspect_deg": float(theta)})
                if best is None or chi2 < best[0]:
                    best = (chi2, params)
    else:
        raise ValueError(f"unknown model {name!r}")
    return grid, best[0], best[1]


_MODEL_K = {"jacobi": 4, "free": 5, "binary": 5}


def _fit_one_model(name, lc, scatterings, aspect_grid, render_kw,
                   refine=True):
    """Grid over all laws, then refine only the winning law."""
    per_law = {law: _grid_stage(name, lc, law, aspect_grid, render_kw)
               for law in scatterings}
    law = min(per_law, key=lambda l: per_law[l][1])
    grid, chi2_best, params = per_law[law]

    if refine:
        chi2_best, params = _refine(name, lc, law, params, render_kw)
    else:
        # re-evaluate the winning node at final resolution for honest chi2
        chi2_best, params = _reeval(name, lc, law, params, render_kw)

    return _finalize(name, lc, law, _MODEL_K[name], chi2_best, params,
                     grid, render_kw)


def _node_curve(name, lc, law, params, n_phases, n_pixels):
    """Render (via cache) the dense curve of a parameter point."""
    if name == "jacobi":
        return models.solo_curve(models.jacobi_axes(params["b_over_a"]),
                                 params["aspect_deg"], lc.alpha_deg, law,
                                 n_phases=n_phases, n_pixels=n_pixels)
    if name == "free":
        axes = (1.0, params["b_over_a"],
                params["b_over_a"] * params["c_over_b"])
        return models.solo_curve(axes, params["aspect_deg"], lc.alpha_deg,
                                 law, n_phases=n_phases, n_pixels=n_pixels)
    return models.duet_curve(params["q"], params["c1"], params["aspect_deg"],
                             lc.alpha_deg, law,
                             n_phases=n_phases, n_pixels=n_pixels)


def _reeval(name, lc, law, params, render_kw):
    """chi2 (and phi0/zp) of a parameter point at final resolution."""
    ph, dm = _node_curve(name, lc, law, params,
                         render_kw["n_phases"], render_kw["n_pixels"])
    chi2, phi0, zp = compare.scan_phi0(lc, ph, dm, polish=True)
    return chi2, dict(params, phi0=phi0, zeropoint=zp)


def _refine(name, lc, law, params, render_kw):
    """Nelder-Mead over shape and aspect from the best grid node.

    The rotational phase offset and zero-point are optimized analytically
    inside the objective (:func:`soloduet.compare.scan_phi0` on the cached
    curve), so the simplex never re-renders to move ``phi0`` — only genuine
    shape/aspect steps cost a render.  The simplex runs at grid resolution
    (its accuracy is limited by the data noise, not the render), and the
    refined point is re-evaluated once at final resolution.
    """
    n_phases = render_kw["grid_n_phases"]
    n_pixels = render_kw["grid_n_pixels"]

    if name == "jacobi":
        names = ["b_over_a", "aspect_deg"]
        bounds = [(BA_MIN, 1.0), (5.0, 90.0)]
    elif name == "free":
        names = ["b_over_a", "c_over_b", "aspect_deg"]
        bounds = [(0.05, 1.0), (0.05, 1.0), (5.0, 90.0)]
    else:
        names = ["c1", "aspect_deg"]
        fams = models.binary_families()
        q_grid = np.array(sorted(fams))
        f = fams[float(q_grid[np.argmin(np.abs(q_grid - params["q"]))])]
        bounds = [(float(f["c1"][0]), float(f["c1"][-1])), (5.0, 90.0)]

    def objective(x):
        if any(not (lo <= v <= hi) for v, (lo, hi) in zip(x, bounds)):
            return 1e12
        p = dict(params, **{nm: float(v) for nm, v in zip(names, x)})
        try:
            ph, dm = _node_curve(name, lc, law, p, n_phases, n_pixels)
        except (ValueError, RuntimeError):
            return 1e12
        chi2, _, _ = compare.scan_phi0(lc, ph, dm, n_scan=32, polish=True)
        return chi2

    x0 = np.array([params[nm] for nm in names])
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"xatol": 5e-4, "fatol": 0.05,
                            "maxfev": 60 * len(names)})
    chi2_x0 = objective(x0)
    x = res.x if res.fun <= chi2_x0 else x0

    new = dict(params, **{nm: float(np.clip(v, lo, hi))
                          for nm, v, (lo, hi) in zip(names, x, bounds)})
    return _reeval(name, lc, law, new, render_kw)


def _finalize(name, lc, law, k, chi2, params, grid, render_kw) -> ModelFit:
    """Assemble the ModelFit (dense curve, density, 1-sigma ranges, notes)."""
    n_phases = render_kw["n_phases"]
    n_pixels = render_kw["n_pixels"]
    notes: List[str] = []
    density = None

    if name == "jacobi":
        curve = models.solo_curve(models.jacobi_axes(params["b_over_a"]),
                                  params["aspect_deg"], lc.alpha_deg, law,
                                  n_phases=n_phases, n_pixels=n_pixels)
        density = density_from_spin(models.jacobi_omega2(params["b_over_a"]),
                                    lc.period_hr)
        fig_axes = models.jacobi_axes(params["b_over_a"])
        params = dict(params, c_over_a=fig_axes[2])
    elif name == "free":
        curve = models.solo_curve(
            (1.0, params["b_over_a"], params["b_over_a"] * params["c_over_b"]),
            params["aspect_deg"], lc.alpha_deg, law,
            n_phases=n_phases, n_pixels=n_pixels)
        notes.append("free-ellipsoid mode: shape is not an equilibrium "
                     "figure, no density can be inferred")
    else:
        prim, sec, dist, omega2, sep = models.binary_scene(
            params["q"], params["c1"])
        curve = models.duet_curve(params["q"], params["c1"],
                                  params["aspect_deg"], lc.alpha_deg, law,
                                  n_phases=n_phases, n_pixels=n_pixels)
        density = density_from_spin(omega2, lc.period_hr)
        params = dict(params, separation=sep, b1=prim[1],
                      secondary_scale=sec[0])
        if sep < 1.02:
            notes.append("components in (or very close to) contact")

    fit = ModelFit(name=name, scattering=law, params=params,
                   chi2=float(chi2), n=len(lc), k=k, curve=curve,
                   density_kgm3=density, grid=grid, notes=notes)
    fit.within_1sig = _within_ranges(grid, fit.chi2)
    return fit


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def fit_lightcurve(
    lc: FoldedLightcurve,
    modes: Sequence[str] = ("jacobi", "binary"),
    free_ellipsoid: bool = False,
    scatterings: Sequence[str] = SCATTERING_LAWS,
    aspect_grid: Sequence[float] = DEFAULT_ASPECT_GRID,
    n_pixels: int = models.DEFAULT_N_PIXELS,
    n_phases: int = models.DEFAULT_N_PHASES,
    grid_n_pixels: int = 96,
    grid_n_phases: int = 64,
    ba_step: float = 0.02,
    q_step: float = 0.05,
    n_c1: int = 8,
    refine: bool = True,
) -> SoloDuetResult:
    """Fit the solo and duet models and deliver the verdict.

    Parameters
    ----------
    lc : FoldedLightcurve
        Folded differential light curve (see :func:`soloduet.io.phase_fold`).
        The fold period is in **hours** and equals one rotation of a single
        body or one orbit of a tidally locked binary.
    modes : sequence of str
        Model classes to fit from {"jacobi", "binary"}; ``free_ellipsoid``
        adds the unconstrained triaxial mode.
    aspect_grid : sequence of float
        Aspect angles [deg] to try (pass a single value if the pole is known).
    """
    modes = list(modes)
    if free_ellipsoid and "free" not in modes:
        modes.append("free")

    render_kw = dict(n_phases=int(n_phases), n_pixels=int(n_pixels),
                     grid_n_phases=int(grid_n_phases),
                     grid_n_pixels=int(grid_n_pixels),
                     ba_step=float(ba_step), q_step=float(q_step),
                     n_c1=int(n_c1))

    fits: Dict[str, ModelFit] = {}
    for name in modes:
        fits[name] = _fit_one_model(name, lc, tuple(scatterings),
                                    tuple(aspect_grid), render_kw,
                                    refine=refine)

    verdict = _verdict(lc, fits)
    return SoloDuetResult(lc=lc, fits=fits, verdict=verdict,
                          period_hr=lc.period_hr)


def _verdict(lc: FoldedLightcurve, fits: Dict[str, ModelFit]) -> Verdict:
    morph = compare.morphology(lc)
    singles = [f for nm, f in fits.items() if nm in ("jacobi", "free") and f]
    binary = fits.get("binary")

    if not singles or binary is None:
        return Verdict(preferred="indeterminate", strength="n/a",
                       delta_bic=0.0, delta_aic=0.0, f_test_p=1.0,
                       morphology=morph,
                       caveats=["both a single and a binary model are needed "
                                "for a verdict"])

    single = min(singles, key=lambda f: f.bic)
    delta_bic = single.bic - binary.bic     # > 0 favors binary
    delta_aic = single.aic - binary.aic
    p = compare.f_test_p(single.chi2, single.dof, binary.chi2, binary.dof)
    strength = compare.bic_strength(delta_bic)

    caveats: List[str] = []
    err_floor = float(np.median(lc.err))
    if morph.amplitude < 4.0 * err_floor:
        preferred = "indeterminate"
        caveats.append("no significant modulation: fitted amplitude "
                       f"({morph.amplitude:.3f} mag) is below 4x the median "
                       f"uncertainty ({err_floor:.3f} mag), so no shape "
                       "verdict is possible")
    elif strength == "indistinguishable":
        preferred = "indeterminate"
        caveats.append("models are statistically indistinguishable from this "
                       "light curve alone (cf. 2000 GN171 in Lacerda & "
                       "Jewitt 2007)")
    else:
        preferred = "binary" if delta_bic > 0 else "single"
    if min(single.redchi2, binary.redchi2) > 3.0:
        caveats.append("neither model describes the data well "
                       "(reduced chi^2 > 3): consider albedo variegation, "
                       "non-principal-axis rotation, or a wrong period")

    if morph.exceeds_single_limit and preferred != "binary":
        caveats.append("morphology contradicts the statistics: range > 0.9 "
                       "mag cannot come from a single equilibrium figure")
    if not morph.low_alpha_valid:
        caveats.append(f"phase angle {lc.alpha_deg:.1f} deg > "
                       f"{compare.MORPHOLOGY_ALPHA_MAX:.0f} deg: V/U minima "
                       "diagnostic unreliable (minima broaden with alpha)")
    if morph.amplitude < 0.15:
        caveats.append("low amplitude: aspect-shape degeneracy is severe; "
                       "a near-pole-on viewing of any shape can mimic this")
    if "free" in fits and fits["free"] is not None:
        caveats.append("free-ellipsoid mode carries no density constraint")

    return Verdict(preferred=preferred, strength=strength,
                   delta_bic=float(delta_bic), delta_aic=float(delta_aic),
                   f_test_p=float(p), morphology=morph, caveats=caveats)


__all__ = [
    "ModelFit",
    "Verdict",
    "SoloDuetResult",
    "fit_lightcurve",
    "GridRecord",
    "DEFAULT_ASPECT_GRID",
    "SCATTERING_LAWS",
]
