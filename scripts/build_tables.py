"""Regenerate the precomputed grids shipped in soloduet/data/.

Usage:  python scripts/build_tables.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soloduet.tables import build_all  # noqa: E402

if __name__ == "__main__":
    build_all(verbose=True)
