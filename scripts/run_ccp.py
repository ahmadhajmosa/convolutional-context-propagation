#!/usr/bin/env python3
"""CLI wrapper for CCP pipeline."""

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ccp.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
