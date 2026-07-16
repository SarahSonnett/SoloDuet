"""Light-curve input for SoloDuet.

Two entry points:

* :func:`read_lightcurve` — reads a tabular photometry file.  Rich,
  header-labelled tables (SpinDoc/Silhouette style, with distances and phase
  angles) are parsed with Silhouette's flexible alias-map reader when
  available; bare ``time mag [merr]`` tables are handled natively.
* :func:`phase_fold` — folds photometry on a rotation period and converts to
  differential magnitudes about the (inverse-variance weighted) mean,
  producing the :class:`FoldedLightcurve` that the fitting layer consumes.

Conventions
-----------
* **Periods are in HOURS** throughout the user-facing API (SoloDuet follows
  the SpotLight/SpinDoc convention; note Silhouette uses days).
* Differential magnitudes are relative to the weighted mean; magnitudes
  increase downward in all plots.
* One full light-curve period corresponds to one **rotation** for a single
  body and one **orbit** for a tidally locked binary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._compat import HAVE_SILHOUETTE

if HAVE_SILHOUETTE:
    from silhouette.io import Photometry, read_photometry  # type: ignore
else:  # pragma: no cover - exercised only when Silhouette is absent
    Photometry = None
    read_photometry = None

HOURS_PER_DAY = 24.0


@dataclass
class FoldedLightcurve:
    """A phase-folded differential light curve.

    Attributes
    ----------
    phase : np.ndarray
        Rotational (or orbital) phase in [0, 1).
    dmag : np.ndarray
        Differential magnitude about the inverse-variance weighted mean.
    err : np.ndarray
        1-sigma magnitude uncertainties.
    period_hr : float
        Fold period in hours.
    alpha_deg : float
        Representative (median) solar phase angle of the data, degrees.
    epoch_mjd : float
        MJD of rotational phase zero.
    object_name : str, optional
        Target designation.
    """

    phase: np.ndarray
    dmag: np.ndarray
    err: np.ndarray
    period_hr: float
    alpha_deg: float = 0.0
    epoch_mjd: Optional[float] = None
    object_name: Optional[str] = None

    def __len__(self) -> int:
        return int(self.phase.size)

    @property
    def amplitude(self) -> float:
        """Peak-to-trough range (mag) of a 10th/90th-percentile-safe estimate."""
        return float(np.max(self.dmag) - np.min(self.dmag))

    def sorted_by_phase(self) -> "FoldedLightcurve":
        order = np.argsort(self.phase)
        return FoldedLightcurve(
            phase=self.phase[order],
            dmag=self.dmag[order],
            err=self.err[order],
            period_hr=self.period_hr,
            alpha_deg=self.alpha_deg,
            epoch_mjd=self.epoch_mjd,
            object_name=self.object_name,
        )


def _read_simple(infile: str):
    """Parse a bare ``time mag [merr]`` whitespace/comma table.

    Lines starting with ``#`` are comments; a non-numeric first row is
    treated as a header and skipped.  JD epochs (~2.4e6) are converted to MJD.
    """
    rows = []
    with open(infile) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(p) for p in parts[:3]])
            except ValueError:
                continue  # header or junk line
    if not rows:
        raise ValueError(f"No numeric data rows found in {infile!r}")
    ncol = min(len(r) for r in rows)
    if ncol < 2:
        raise ValueError(f"{infile!r} needs at least two columns (time, mag)")
    arr = np.array([r[:ncol] for r in rows], dtype=float)
    time, mag = arr[:, 0], arr[:, 1]
    merr = arr[:, 2] if ncol >= 3 else np.full_like(mag, 0.02)
    if np.nanmedian(time) > 2_400_000.0:  # JD -> MJD
        time = time - 2_400_000.5
    return time, mag, merr


def read_lightcurve(
    infile: str,
    object_name: Optional[str] = None,
):
    """Read a photometry file, returning ``(time_mjd, mag, merr, alpha_deg)``.

    Rich tables with recognised headers (``MJD/JD``, ``mag``, ``merr``,
    ``alpha`` …) are parsed with Silhouette's reader when available, and the
    magnitudes are reduced to unit heliocentric/geocentric distance
    (``m - 5 log10(r * delta)``) so multi-night data phase together cleanly.
    Bare ``time mag [merr]`` tables are read directly and returned as-is with
    ``alpha_deg = 0``.
    """
    if HAVE_SILHOUETTE and read_photometry is not None:
        try:
            phot = read_photometry(infile, object_name=object_name)
            red_mag = phot.mag - 5.0 * np.log10(phot.rhelio * phot.delta)
            alpha = float(np.nanmedian(phot.alpha))
            return phot.time, red_mag, phot.merr, alpha
        except (ValueError, KeyError):
            pass  # fall through to the simple parser
    time, mag, merr = _read_simple(infile)
    return time, mag, merr, 0.0


def hg_phase_correction(alpha_deg, G: float = 0.15) -> np.ndarray:
    """Solar-phase dimming [mag] predicted by the IAU H-G model at ``alpha``.

    Returns ``-2.5 log10[(1-G) Phi_1 + G Phi_2]`` (>= 0, zero at opposition;
    Bowell et al. 1989), computed with SpinDoc's ``HGfunction`` when the
    sibling repo is importable.  Subtract this from distance-reduced
    magnitudes to reference every epoch to zero phase angle, so nights at
    different solar phase angles synchronize when folded together:

        m_sync = m - 5 log10(r * delta) - hg_phase_correction(alpha, G)

    This aligns the *mean levels* of different epochs; the (second-order)
    change of light-curve shape with phase angle is not corrected, so quote
    the median alpha of the combined data to the fitter.
    """
    from ._compat import HGfunction

    alpha_deg = np.asarray(alpha_deg, dtype=float)
    return HGfunction(alpha_deg, 0.0, float(G))


def reduce_and_correct(phot, G: float = 0.15) -> np.ndarray:
    """Distance-reduced, phase-corrected magnitudes of a ``Photometry`` table.

    ``m - 5 log10(rhelio * delta) - hg_phase_correction(alpha, G)`` — the
    standard reduction for combining multi-epoch photometry before folding.
    """
    return (phot.mag - 5.0 * np.log10(phot.rhelio * phot.delta)
            - hg_phase_correction(phot.alpha, G))


def phase_fold(
    time_mjd: np.ndarray,
    mag: np.ndarray,
    merr: np.ndarray,
    period_hr: float,
    alpha_deg: float = 0.0,
    epoch_mjd: Optional[float] = None,
    object_name: Optional[str] = None,
) -> FoldedLightcurve:
    """Fold photometry on ``period_hr`` (HOURS) into a :class:`FoldedLightcurve`.

    Differential magnitudes are computed about the inverse-variance weighted
    mean, so pre-mean-subtracted input is handled identically.
    """
    time_mjd = np.asarray(time_mjd, dtype=float)
    mag = np.asarray(mag, dtype=float)
    merr = np.asarray(merr, dtype=float)
    if epoch_mjd is None:
        epoch_mjd = float(np.min(time_mjd))
    period_days = period_hr / HOURS_PER_DAY
    phase = np.mod((time_mjd - epoch_mjd) / period_days, 1.0)
    w = 1.0 / np.clip(merr, 1e-6, None) ** 2
    dmag = mag - np.sum(w * mag) / np.sum(w)
    return FoldedLightcurve(
        phase=phase,
        dmag=dmag,
        err=merr,
        period_hr=float(period_hr),
        alpha_deg=float(alpha_deg),
        epoch_mjd=epoch_mjd,
        object_name=object_name,
    ).sorted_by_phase()


__all__ = [
    "FoldedLightcurve",
    "read_lightcurve",
    "phase_fold",
    "hg_phase_correction",
    "reduce_and_correct",
    "Photometry",
    "read_photometry",
]
