#!/usr/bin/env python
"""Entry point for RL Bot.

Kept as a one-liner on purpose: the algorithm lives in the importable ``rlbot``
package so it can be tested and reused. Run ``python main.py --help`` for the
full command line.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rlbot.cli import main  # noqa: E402 - needs the path fix above

if __name__ == "__main__":
    raise SystemExit(main())
