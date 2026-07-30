import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlbot.config import Config  # noqa: E402
from rlbot.environment import GridWorld  # noqa: E402


@pytest.fixture
def config() -> Config:
    """Small, fast, fully deterministic training config."""
    return Config(grid_size=4, goal=(3, 3), episodes=40, max_steps=60, seed=1234)


@pytest.fixture
def env(config: Config) -> GridWorld:
    return GridWorld(config)
