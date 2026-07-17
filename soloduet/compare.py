"""Goodness-of-fit, information criteria, and morphology diagnostics.

Statistics
----------
Model curves are compared to the folded data with chi^2; the rotational
phase offset is scanned and the magnitude zero-point solved analytically at
every evaluation.  Model classes (single ellipsoid vs. Roche binary) are
compared with Delta-BIC on the Kass & Raftery (1995) evidence ladder,
corroborated by Delta-AIC and a heuristic F-test (the models are not nested,
so the F-test p-value is indicative only).

Morphology
----------
Independent, physics-based diagnostics from the literature:

* peak-to-peak range above **0.9 mag** cannot be produced by any single
  equilibrium figure under geometric scattering and requires a contact
  binary or albedo variegation (Weidenschilling 1980; Leone et al. 1984);
* contact binaries show **V-shaped minima and rounded (inverted-U) maxima**
  near zero solar phase angle, while ellipsoids give near-sinusoidal curves
  — quantified here by the curvature ratio of a Fourier-series fit at the
  extrema.  The diagnostic is flagged unreliable at phase angles above a few
  degrees, where binary minima broaden (Lacerda & Jewitt 2007);
* unequal minima depths (odd Fourier harmonics) indicate asymmetry, common
  for unequal-component binaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .io import FoldedLightcurve

#: maximum peak-to-peak range of a single equilibrium figure [mag]
SINGLE_FIGURE_LIMIT = 0.9

#: phase angle [deg] above which the V/U minima diagnostic is unreliable
MORPHOLOGY_ALPHA_MAX = 5.0


# ---------------------------------------------------------------------------
# chi^2 machinery
# ---------------------------------------------------------------------------

def interp_periodic(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linear interpolation of a periodic (period-1) tabulated curve."""
    xp_ext = np.concatenate([xp, [xp[0] + 1.0]])
    fp_ext = np.concatenate([fp, [fp[0]]])
    return np.interp(np.mod(x, 1.0), xp_ext, fp_ext)


def chi2_with_zeropoint(lc: FoldedLightcurve, model_phase: np.ndarray,
                        model_dmag: np.ndarray, phi0: float) -> Tuple[float, float]:
    """chi^2 of the model shifted by ``phi0``, with the analytic zero-point.

    The zero-point (inverse-variance weighted mean residual) makes the
    result independent of how either curve was mean-referenced.
    """
    m = interp_periodic(lc.phase - phi0, model_phase, model_dmag)
    w = 1.0 / np.clip(lc.err, 1e-6, None) ** 2
    resid = lc.dmag - m
    zp = float(np.sum(w * resid) / np.sum(w))
    chi2 = float(np.sum(w * (resid - zp) ** 2))
    return chi2, zp


def scan_phi0(lc: FoldedLightcurve, model_phase: np.ndarray,
              model_dmag: np.ndarray, n_scan: int = 64,
              polish: bool = True) -> Tuple[float, float, float]:
    """Best ``(chi2, phi0, zp)`` over the phase offset.

    Uniform coarse scan (which also catches the Delta-phi0 = 0.5 alias of
    near-symmetric curves), optionally polished with a bounded 1-D
    minimization around the best node.  No rendering happens here — the
    model curve is fixed — so this is cheap enough to serve as the inner
    loop of the shape/aspect optimizer.
    """
    best = (np.inf, 0.0, 0.0)
    for phi0 in np.arange(n_scan) / float(n_scan):
        chi2, zp = chi2_with_zeropoint(lc, model_phase, model_dmag, float(phi0))
        if chi2 < best[0]:
            best = (chi2, float(phi0), zp)
    if polish:
        from scipy.optimize import minimize_scalar

        span = 1.0 / n_scan
        res = minimize_scalar(
            lambda p: chi2_with_zeropoint(lc, model_phase, model_dmag, p)[0],
            bounds=(best[1] - span, best[1] + span), method="bounded",
            options={"xatol": 1e-5})
        if res.fun < best[0]:
            chi2, zp = chi2_with_zeropoint(lc, model_phase, model_dmag,
                                           float(res.x))
            best = (chi2, float(np.mod(res.x, 1.0)), zp)
    return best


def aic(chi2: float, k: int, n: int) -> float:
    """Akaike information criterion (Gaussian likelihood)."""
    return chi2 + 2.0 * k


def bic(chi2: float, k: int, n: int) -> float:
    """Bayesian information criterion (Schwarz 1978, Gaussian likelihood)."""
    return chi2 + k * np.log(n)


def f_test_p(chi2_a: float, dof_a: int, chi2_b: float, dof_b: int) -> float:
    """Two-sided p-value that the reduced chi^2 of A and B differ (F-test).

    Heuristic for non-nested models — reported for context only; the verdict
    rests on Delta-BIC.
    """
    from scipy.stats import f as f_dist

    ra = chi2_a / max(dof_a, 1)
    rb = chi2_b / max(dof_b, 1)
    if ra <= 0.0 or rb <= 0.0:
        return 1.0
    if ra >= rb:
        stat, d1, d2 = ra / rb, dof_a, dof_b
    else:
        stat, d1, d2 = rb / ra, dof_b, dof_a
    p = 2.0 * min(f_dist.sf(stat, d1, d2), 1.0 - f_dist.sf(stat, d1, d2))
    return float(np.clip(p, 0.0, 1.0))


def bic_strength(delta_bic: float) -> str:
    """Kass & Raftery (1995) evidence ladder for |Delta BIC| (auxiliary)."""
    d = abs(delta_bic)
    if d < 2.0:
        return "indistinguishable"
    if d < 6.0:
        return "positive"
    if d < 10.0:
        return "strong"
    return "very strong"


#: reduced chi^2 below which a model is considered an adequate description
ADEQUATE_REDCHI2 = 3.0

#: chi^2_nu ratio thresholds of the conservative evidence ladder
RATIO_CLAIM = 1.5      # below this: no preference is claimed at all
RATIO_MODERATE = 2.0
RATIO_STRONG = 3.0
RATIO_VERY_STRONG = 5.0


def chi2_evidence(redchi2_winner: float, redchi2_loser: float) -> str:
    """Conservative evidence label from the reduced-chi^2 comparison.

    Evidence language is earned by *fit quality*, not by information-
    criterion arithmetic: similar chi^2_nu values mean the models cannot be
    told apart no matter how many data points sharpen the formal
    statistics.  The ladder:

    * ratio < 1.5 — ``"indistinguishable"`` (no preference claimed, even if
      the two values happen to straddle the adequacy threshold);
    * ratio < 2 — ``"weak"`` unless the winner is adequate
      (chi^2_nu < 3) while the loser is not, which earns ``"moderate"``;
    * ratio >= 2 — ``"moderate"``;
    * ratio >= 3 — ``"strong"``;
    * ratio >= 5 *and* the winner adequate — ``"very strong"``.
    """
    ratio = redchi2_loser / max(redchi2_winner, 1e-12)
    adequacy_split = (redchi2_winner < ADEQUATE_REDCHI2 <= redchi2_loser)
    if ratio < RATIO_CLAIM:
        return "indistinguishable"
    if ratio >= RATIO_VERY_STRONG and redchi2_winner < ADEQUATE_REDCHI2:
        return "very strong"
    if ratio >= RATIO_STRONG:
        return "strong"
    if ratio >= RATIO_MODERATE or adequacy_split:
        return "moderate"
    return "weak"


# ---------------------------------------------------------------------------
# morphology diagnostics
# ---------------------------------------------------------------------------

@dataclass
class MorphologyReport:
    """Model-independent light-curve shape diagnostics."""

    amplitude: float                  # robust peak-to-peak range [mag]
    exceeds_single_limit: bool        # amplitude > 0.9 mag
    minima_curvature_ratio: float     # |d2m/dphase2| minima / maxima
    minima_depth_difference: float    # |depth1 - depth2| [mag]
    odd_harmonic_fraction: float      # odd-k Fourier power fraction
    low_alpha_valid: bool             # alpha low enough for the V/U metric
    notes: List[str] = field(default_factory=list)

    def summary_lines(self) -> List[str]:
        out = [
            f"amplitude              : {self.amplitude:.3f} mag"
            + (" (> 0.9 mag single-figure limit!)" if self.exceeds_single_limit else ""),
            f"minima/maxima curvature: {self.minima_curvature_ratio:.2f}"
            + ("" if self.low_alpha_valid else "  [unreliable: alpha too large]"),
            f"minima depth difference: {self.minima_depth_difference:.3f} mag",
            f"odd-harmonic power     : {100 * self.odd_harmonic_fraction:.1f} %",
        ]
        return out + self.notes


def _fourier_design(phase: np.ndarray, order: int) -> np.ndarray:
    cols = [np.ones_like(phase)]
    for k in range(1, order + 1):
        cols.append(np.cos(2.0 * np.pi * k * phase))
        cols.append(np.sin(2.0 * np.pi * k * phase))
    return np.column_stack(cols)


def _fourier_eval(coeff: np.ndarray, phase: np.ndarray, order: int,
                  deriv: int = 0) -> np.ndarray:
    out = np.zeros_like(phase) if deriv else np.full_like(phase, coeff[0])
    for k in range(1, order + 1):
        a, b = coeff[2 * k - 1], coeff[2 * k]
        w = 2.0 * np.pi * k
        if deriv == 0:
            out += a * np.cos(w * phase) + b * np.sin(w * phase)
        elif deriv == 2:
            out += -w * w * (a * np.cos(w * phase) + b * np.sin(w * phase))
    return out


def morphology(lc: FoldedLightcurve, order: int = 4) -> MorphologyReport:
    """Compute the :class:`MorphologyReport` from a Fourier fit to the data."""
    w = 1.0 / np.clip(lc.err, 1e-6, None)
    A = _fourier_design(lc.phase, order) * w[:, None]
    coeff, *_ = np.linalg.lstsq(A, lc.dmag * w, rcond=None)

    dense = np.linspace(0.0, 1.0, 512, endpoint=False)
    m = _fourier_eval(coeff, dense, order)
    m2 = _fourier_eval(coeff, dense, order, deriv=2)
    amplitude = float(m.max() - m.min())

    # extrema of the smooth model (mag increases downward: minima of
    # brightness = maxima of dmag)
    faint_mask = m > np.percentile(m, 80)
    bright_mask = m < np.percentile(m, 20)
    curv_min = float(np.mean(np.abs(m2[faint_mask])))
    curv_max = float(np.mean(np.abs(m2[bright_mask])))
    curvature_ratio = curv_min / curv_max if curv_max > 0 else np.inf

    # depths of the two brightness minima (halves of the folded curve)
    half1, half2 = dense < 0.5, dense >= 0.5
    depth_diff = float(abs(m[half1].max() - m[half2].max())) \
        if half1.any() and half2.any() else 0.0

    power = coeff[1:] ** 2
    k_idx = np.repeat(np.arange(1, order + 1), 2)
    total = float(power.sum())
    odd_frac = float(power[k_idx % 2 == 1].sum() / total) if total > 0 else 0.0

    notes = []
    if amplitude > SINGLE_FIGURE_LIMIT:
        notes.append(
            "range > 0.9 mag exceeds the maximum of a single equilibrium "
            "figure (Weidenschilling 1980; Leone et al. 1984)")
    return MorphologyReport(
        amplitude=amplitude,
        exceeds_single_limit=amplitude > SINGLE_FIGURE_LIMIT,
        minima_curvature_ratio=curvature_ratio,
        minima_depth_difference=depth_diff,
        odd_harmonic_fraction=odd_frac,
        low_alpha_valid=lc.alpha_deg <= MORPHOLOGY_ALPHA_MAX,
        notes=notes,
    )


__all__ = [
    "chi2_with_zeropoint",
    "scan_phi0",
    "interp_periodic",
    "aic",
    "bic",
    "f_test_p",
    "bic_strength",
    "chi2_evidence",
    "ADEQUATE_REDCHI2",
    "RATIO_CLAIM",
    "RATIO_MODERATE",
    "RATIO_STRONG",
    "RATIO_VERY_STRONG",
    "MorphologyReport",
    "morphology",
    "SINGLE_FIGURE_LIMIT",
    "MORPHOLOGY_ALPHA_MAX",
]
