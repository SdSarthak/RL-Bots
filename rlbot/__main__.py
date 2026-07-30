"""Allow ``python -m rlbot ...``."""

from rlbot.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
