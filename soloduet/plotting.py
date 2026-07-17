"""Publication-style summary figures (Agg backend, GridSpec layout).

The main deliverable is :func:`save_summary`: a three-row figure with the
phased data and both best-fit model curves (+ residuals), a rendered shape
mosaic of the two competing models, and the chi^2 landscapes with the
verdict panel.
"""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402

from . import models  # noqa: E402
from .compare import interp_periodic  # noqa: E402
from .geometry import view_vectors  # noqa: E402
from .render import binary_bodies, render, solo_bodies  # noqa: E402

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.titlesize": 13,
})

_MODEL_STYLE = {
    "jacobi": dict(color="#1f77b4", label="Jacobi ellipsoid (solo)"),
    "free": dict(color="#2ca02c", label="free ellipsoid (solo)"),
    "binary": dict(color="#d62728", label="Roche binary (duet)"),
}

_MOSAIC_PHASES = (0.00, 0.15, 0.25, 0.40)


def _model_on(fit, phase_grid):
    """Best-fit model dmag evaluated at data phases (phi0 + zero-point)."""
    mph, mdm = fit.curve
    return interp_periodic(phase_grid - fit.params["phi0"], mph, mdm) \
        + fit.params["zeropoint"]


def _bodies_for(fit):
    if fit.name == "binary":
        prim, sec, dist, _, _ = models.binary_scene(fit.params["q"],
                                                    fit.params["c1"])
        return binary_bodies(prim, sec, dist)
    if fit.name == "jacobi":
        return solo_bodies(models.jacobi_axes(fit.params["b_over_a"]))
    ba = fit.params["b_over_a"]
    return solo_bodies((1.0, ba, ba * fit.params["c_over_b"]))


def plot_lightcurve_fit(result, ax, ax_resid=None):
    """Phased data with every fitted model curve overplotted."""
    lc = result.lc
    ax.errorbar(lc.phase, lc.dmag, yerr=lc.err, fmt="o", ms=3.5,
                color="0.25", ecolor="0.6", elinewidth=0.8, zorder=3,
                label="data")
    dense = np.linspace(0, 1, 400, endpoint=False)
    for name, fit in result.fits.items():
        st = _MODEL_STYLE[name]
        m = _model_on(fit, dense)
        ax.plot(dense, m, "-" if name == "binary" else "--", lw=1.6,
                color=st["color"],
                label=st["label"]
                + r"  ($\chi^2_\nu$ = " + f"{fit.redchi2:.2f})")
        if ax_resid is not None:
            r = lc.dmag - _model_on(fit, lc.phase)
            ax_resid.plot(lc.phase, r, ".", ms=3, color=st["color"], alpha=0.8)
    ax.invert_yaxis()
    ax.set_ylabel("differential magnitude")
    ax.legend(loc="lower center", ncol=3, fontsize=8, frameon=False)
    ax.set_xlim(0, 1)
    if ax_resid is not None:
        ax_resid.axhline(0.0, color="0.7", lw=0.8)
        ax_resid.invert_yaxis()
        ax_resid.set_xlim(0, 1)
        ax_resid.set_ylabel("resid")
        ax_resid.set_xlabel("rotational phase")
    else:
        ax.set_xlabel("rotational phase")


def plot_shape_mosaic(result, subspec, fig):
    """Rendered images of the best solo (top) and duet (bottom) models."""
    rows = [nm for nm in ("jacobi", "free", "binary") if nm in result.fits]
    # show at most one solo (the better one) plus the binary
    solos = [nm for nm in rows if nm != "binary"]
    if len(solos) > 1:
        best_solo = min(solos, key=lambda nm: result.fits[nm].bic)
        rows = [best_solo] + (["binary"] if "binary" in rows else [])
    inner = GridSpecFromSubplotSpec(len(rows), len(_MOSAIC_PHASES),
                                    subplot_spec=subspec, wspace=0.02,
                                    hspace=0.05)
    lc = result.lc
    for i, nm in enumerate(rows):
        fit = result.fits[nm]
        bodies = _bodies_for(fit)
        for j, ph in enumerate(_MOSAIC_PHASES):
            axi = fig.add_subplot(inner[i, j])
            obs, sun = view_vectors(fit.params["aspect_deg"], lc.alpha_deg,
                                    ph)
            _, img = render(bodies, obs, sun, n_pixels=160,
                            scattering=fit.scattering, return_image=True)
            axi.imshow(img, cmap="gray", origin="lower")
            axi.set_xticks([])
            axi.set_yticks([])
            if i == 0:
                axi.set_title(f"phase {ph:.2f}", fontsize=8)
            if j == 0:
                lab = {"jacobi": "Jacobi solo", "free": "free solo",
                       "binary": "Roche duet"}[nm]
                axi.set_ylabel(lab, fontsize=8)


def plot_chi2_landscapes(result, ax_solo, ax_duet):
    """chi^2/chi^2_best landscapes from the grid stage."""
    # solo: chi2 vs b/a (min over aspect)
    for nm in ("jacobi", "free"):
        fit = result.fits.get(nm)
        if fit is None or not fit.grid:
            continue
        ba = np.array([g.params["b_over_a"] for g in fit.grid])
        chi = np.array([g.chi2 for g in fit.grid])
        best = np.array([chi[ba == v].min() for v in np.unique(ba)])
        ax_solo.plot(np.unique(ba), best / chi.min(), "o-", ms=3,
                     color=_MODEL_STYLE[nm]["color"], lw=1)
    ax_solo.axhline(2.0, color="0.5", ls=":", lw=1)
    ax_solo.set_yscale("log")
    ax_solo.set_xlabel("b/a")
    ax_solo.set_ylabel(r"$\chi^2/\chi^2_{\rm best}$")
    ax_solo.set_title("solo: shape sequence (min over aspect)", fontsize=9)

    fit = result.fits.get("binary")
    if fit is not None and fit.grid:
        q = np.array([g.params["q"] for g in fit.grid])
        c1 = np.array([g.params["c1"] for g in fit.grid])
        chi = np.array([g.chi2 for g in fit.grid])
        # min over aspect per (q, c1)
        pts = {}
        for qi, ci, xi in zip(q, c1, chi):
            key = (qi, ci)
            pts[key] = min(pts.get(key, np.inf), xi)
        qq = np.array([k[0] for k in pts])
        cc = np.array([k[1] for k in pts])
        vv = np.array(list(pts.values())) / chi.min()
        sc = ax_duet.scatter(qq, cc, c=np.log10(vv), s=22, cmap="viridis_r",
                             edgecolors="none")
        plt.colorbar(sc, ax=ax_duet, label=r"$\log_{10}\chi^2/\chi^2_{\rm best}$")
        bp = fit.params
        ax_duet.plot(bp["q"], bp["c1"], "r*", ms=12, mec="k", mew=0.5)
    ax_duet.set_xlabel("mass ratio q")
    ax_duet.set_ylabel("primary c/a")
    ax_duet.set_title("duet: Roche grid", fontsize=9)


def plot_verdict_panel(result, ax):
    """Text panel with the verdict and headline numbers (mathtext Greek)."""
    ax.axis("off")
    v = result.verdict
    lines = [f"VERDICT: {v.preferred.upper()}",
             f"evidence: {v.strength}   "
             + r"($\chi^2_\nu$ ratio = " + f"{v.redchi2_ratio:.2f})",
             r"$\Delta$BIC = " + f"{v.delta_bic:+.1f} (auxiliary)"]
    if v.chi2_scale > 1.0:
        lines.append(r"(errors rescaled by $\chi^2_\nu$ = "
                     + f"{v.chi2_scale:.2f})")
    lines.append("")
    for nm in ("jacobi", "free", "binary"):
        fit = result.fits.get(nm)
        if fit is None:
            continue
        if fit.density_kgm3 is not None:
            geq = r"$\geq$ " if fit.density_is_minimum else ""
            rho = (r"$\rho$ " + geq
                   + f"{fit.density_kgm3:.0f} " + r"kg/m$^3$")
        else:
            rho = "no density (free mode)"
        lines.append(f"{nm}: " + r"$\chi^2_\nu$ = "
                     + f"{fit.redchi2:.2f}, " + rho)
        if nm == "binary":
            lines.append(f"   q = {fit.params['q']:.2f}, "
                         + r"$d/(a_1{+}a_2)$ = "
                         + f"{fit.params['separation']:.2f}")
    if any(f.density_is_minimum and f.density_kgm3 is not None
           for f in result.fits.values()):
        lines.append(r"densities are minima (aspect unconstrained)")
    lines.append("")
    lines.append(f"amplitude {v.morphology.amplitude:.2f} mag"
                 + ("  (> 0.9 single-figure limit)"
                    if v.morphology.exceeds_single_limit else ""))
    for cav in v.caveats[:3]:
        lines.append("caveat: " + (cav if len(cav) < 58 else cav[:55] + "..."))
    ax.text(0.02, 0.97, "\n".join(lines), va="top", ha="left",
            transform=ax.transAxes, fontsize=8.5, family="monospace")


def plot_summary(result, figsize=(12.5, 11.0)) -> "matplotlib.figure.Figure":
    """Assemble the full three-row summary figure."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 3, figure=fig, height_ratios=[2.4, 0.7, 1.9, 1.7],
                  hspace=0.45, wspace=0.35)

    ax_lc = fig.add_subplot(gs[0, :])
    ax_res = fig.add_subplot(gs[1, :], sharex=ax_lc)
    plot_lightcurve_fit(result, ax_lc, ax_res)

    plot_shape_mosaic(result, gs[2, :], fig)

    ax_solo = fig.add_subplot(gs[3, 0])
    ax_duet = fig.add_subplot(gs[3, 1])
    ax_txt = fig.add_subplot(gs[3, 2])
    plot_chi2_landscapes(result, ax_solo, ax_duet)
    plot_verdict_panel(result, ax_txt)

    title = "SoloDuet"
    if result.lc.object_name:
        title += f" — {result.lc.object_name}"
    title += f"   (P = {result.period_hr:.4f} h)"
    fig.suptitle(title)
    return fig


def save_summary(result, path: str, dpi: int = 150,
                 figsize=(12.5, 11.0)) -> None:
    """Render and write the summary figure, then release it."""
    fig = plot_summary(result, figsize=figsize)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "plot_summary",
    "save_summary",
    "plot_lightcurve_fit",
    "plot_shape_mosaic",
    "plot_chi2_landscapes",
    "plot_verdict_panel",
]
