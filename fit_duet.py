#!/usr/bin/env python
"""SoloDuet command-line entry point (see ``python fit_duet.py --help``)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soloduet.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
