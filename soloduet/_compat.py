"""Optional coupling to the sibling SpotLight, SpinDoc, and Silhouette repos.

SoloDuet *reuses* code from its sibling projects when they are importable:

* **SpotLight** (``spotlight``) — the forward triaxial-ellipsoid renderer,
  used to cross-validate SoloDuet's own two-body renderer.
* **SpinDoc** (``spindoc``) — the Fourier light-curve model, used for the
  odd-harmonic morphology diagnostic.
* **Silhouette** (``silhouette``) — the flexible tabular photometry reader.

Import strategy ("import both, fall back"):

1. Try a normal ``import``.
2. If that fails, add the sibling repo directory (``~/Projects/<Repo>``,
   assuming the standard layout) to ``sys.path`` and retry.
3. If it is *still* unavailable, fall back to a minimal vendored copy of the
   small pieces SoloDuet actually needs, so the package always runs
   standalone.

The booleans ``HAVE_SPOTLIGHT`` / ``HAVE_SPINDOC`` / ``HAVE_SILHOUETTE``
record what was found.
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECTS = os.path.dirname(os.path.dirname(_HERE))  # …/Projects


def _try_import(modname: str, sibling_dir: str):
    """Import ``modname``, adding ``<Projects>/<sibling_dir>`` to the path if needed."""
    try:
        return importlib.import_module(modname)
    except ImportError:
        cand = os.path.join(_PROJECTS, sibling_dir)
        if os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
        try:
            return importlib.import_module(modname)
        except ImportError:
            return None


spotlight = _try_import("spotlight", "SpotLight")
spindoc = _try_import("spindoc", "SpinDoc")
silhouette = _try_import("silhouette", "Silhouette")

HAVE_SPOTLIGHT = spotlight is not None
HAVE_SPINDOC = spindoc is not None
HAVE_SILHOUETTE = silhouette is not None


# ---------------------------------------------------------------------------
# SpinDoc pieces (with vendored fallbacks)
# ---------------------------------------------------------------------------

if HAVE_SPINDOC:
    from spindoc import fourier  # type: ignore
else:  # pragma: no cover - exercised only when SpinDoc is absent

    def fourier(phase, *coeff):
        """Vendored copy of ``spindoc.fourier`` — Nth-order Fourier series.

        coeff layout: ``[period, mean, phi_1, A_1, phi_2, A_2, ...]``.
        """
        period = coeff[0]
        omega = 2.0 * np.pi / period
        ret = coeff[1] + coeff[3] * np.sin(omega * phase + coeff[2])
        nord = int((len(coeff) - 2) / 2)
        i = 3
        for iord in range(nord - 1):
            ret += coeff[i + 1] * np.sin((iord + 2) * omega * phase + coeff[i + 2])
            i += 2
        return ret


__all__ = [
    "spotlight",
    "spindoc",
    "silhouette",
    "HAVE_SPOTLIGHT",
    "HAVE_SPINDOC",
    "HAVE_SILHOUETTE",
    "fourier",
]
