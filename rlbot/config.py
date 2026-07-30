"""Configuration for the RL Bot grid world and Q-learning agents.

Every tunable lives on :class:`Config`. Defaults reproduce the original
6x6 open-grid setup; JSON files loaded with :meth:`Config.from_file`
override any subset of the fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

Position = Tuple[int, int]

# Movement actions expressed as (d_row, d_col).
ACTIONS: Tuple[Position, ...] = (
    (0, 1),   # Right
    (0, -1),  # Left
    (1, 0),   # Down
    (-1, 0),  # Up
)

ACTION_NAMES: Tuple[str, ...] = ("Right", "Left", "Down", "Up")

# Rendering palette, kept out of Config because it is presentation only.
COLORS: Dict[str, Tuple[int, int, int]] = {
    "WHITE": (255, 255, 255),
    "GRID": (210, 210, 210),
    "RED_AGENT": (200, 0, 0),
    "BLUE_AGENT": (0, 0, 200),
    "RED_PATH": (255, 200, 200),
    "BLUE_PATH": (200, 200, 255),
    "GOAL": (0, 200, 0),
    "OBSTACLE": (70, 70, 70),
    "BLACK": (0, 0, 0),
}


class ConfigError(ValueError):
    """Raised when a configuration is malformed or internally inconsistent."""


def _as_position(value: Any, label: str) -> Position:
    """Coerce ``value`` into a ``(row, col)`` tuple of ints."""
    if isinstance(value, dict):
        try:
            value = (value["row"], value["col"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise ConfigError(f"{label} mapping needs 'row' and 'col' keys") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{label} must be a two-element sequence, got {value!r}")
    if len(value) != 2:
        raise ConfigError(f"{label} must have exactly two elements, got {value!r}")
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must contain integers, got {value!r}") from exc


def _as_positions(values: Iterable[Any], label: str) -> Tuple[Position, ...]:
    return tuple(_as_position(v, label) for v in values)


@dataclass
class Config:
    """All hyperparameters and environment settings for a run."""

    # -------- environment --------
    grid_size: int = 6
    red_start: Position = (0, 0)
    blue_start: Position = (0, 0)
    goal: Position = (5, 5)
    obstacles: Tuple[Position, ...] = ()

    # -------- rewards --------
    goal_reward: float = 100.0
    step_penalty: float = -1.0
    revisit_penalty: float = -5.0
    loop_penalty: float = -10.0

    # -------- Q-learning --------
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.92
    epsilon_min: float = 0.02

    # -------- training loop --------
    episodes: int = 80
    max_steps: int = 200
    no_progress_threshold: int = 10
    efficient_solution_threshold: int = 50
    max_test_steps: int = 100
    seed: int = 0

    # -------- replay buffer --------
    replay_buffer_size: int = 8
    replay_frequency: int = 10
    replay_sample_size: int = 3
    replay_alpha_scale: float = 0.5

    # -------- presentation --------
    cell_size: int = 50
    fps: int = 20
    show_training: bool = True
    print_episode_stats: bool = True
    print_replay_stats: bool = True

    def __post_init__(self) -> None:
        self.red_start = _as_position(self.red_start, "red_start")
        self.blue_start = _as_position(self.blue_start, "blue_start")
        self.goal = _as_position(self.goal, "goal")
        self.obstacles = _as_positions(self.obstacles, "obstacles")
        self.validate()

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def _in_bounds(self, pos: Position) -> bool:
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def validate(self) -> "Config":
        """Check internal consistency, raising :class:`ConfigError` on problems."""
        if self.grid_size < 2:
            raise ConfigError(f"grid_size must be at least 2, got {self.grid_size}")

        for label, pos in (
            ("red_start", self.red_start),
            ("blue_start", self.blue_start),
            ("goal", self.goal),
        ):
            if not self._in_bounds(pos):
                raise ConfigError(
                    f"{label} {pos} is outside a {self.grid_size}x{self.grid_size} grid"
                )

        obstacle_set = set(self.obstacles)
        for pos in obstacle_set:
            if not self._in_bounds(pos):
                raise ConfigError(
                    f"obstacle {pos} is outside a {self.grid_size}x{self.grid_size} grid"
                )
        for label, pos in (
            ("red_start", self.red_start),
            ("blue_start", self.blue_start),
            ("goal", self.goal),
        ):
            if pos in obstacle_set:
                raise ConfigError(f"{label} {pos} sits on an obstacle")

        if self.red_start == self.goal or self.blue_start == self.goal:
            raise ConfigError("start position must differ from the goal")

        if not 0.0 < self.alpha <= 1.0:
            raise ConfigError(f"alpha must be in (0, 1], got {self.alpha}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ConfigError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ConfigError(
                "epsilon values must satisfy 0 <= epsilon_min <= epsilon_start <= 1, "
                f"got min={self.epsilon_min} start={self.epsilon_start}"
            )
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ConfigError(f"epsilon_decay must be in (0, 1], got {self.epsilon_decay}")

        if self.episodes < 1:
            raise ConfigError(f"episodes must be positive, got {self.episodes}")
        if self.max_steps < 1:
            raise ConfigError(f"max_steps must be positive, got {self.max_steps}")
        if self.max_test_steps < 1:
            raise ConfigError(f"max_test_steps must be positive, got {self.max_test_steps}")
        if self.replay_buffer_size < 1:
            raise ConfigError(
                f"replay_buffer_size must be positive, got {self.replay_buffer_size}"
            )
        if self.replay_frequency < 1:
            raise ConfigError(
                f"replay_frequency must be positive, got {self.replay_frequency}"
            )
        if self.replay_sample_size < 1:
            raise ConfigError(
                f"replay_sample_size must be positive, got {self.replay_sample_size}"
            )
        if self.cell_size < 1:
            raise ConfigError(f"cell_size must be positive, got {self.cell_size}")
        if self.fps < 1:
            raise ConfigError(f"fps must be positive, got {self.fps}")
        return self

    # ------------------------------------------------------------------
    # (de)serialisation
    # ------------------------------------------------------------------
    @classmethod
    def field_names(cls) -> List[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build a config from a mapping, ignoring nothing and rejecting typos."""
        known = set(cls.field_names())
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(
                "unknown configuration key(s): " + ", ".join(unknown) +
                ". Valid keys: " + ", ".join(sorted(known))
            )
        return cls(**data)

    @classmethod
    def from_file(cls, path: "str | Path") -> "Config":
        """Load a JSON config file. Missing keys fall back to the defaults."""
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read config file {path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{path} must contain a JSON object at the top level")
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["red_start"] = list(self.red_start)
        data["blue_start"] = list(self.blue_start)
        data["goal"] = list(self.goal)
        data["obstacles"] = [list(o) for o in self.obstacles]
        return data

    def replace(self, **overrides: Any) -> "Config":
        """Return a new config with ``overrides`` applied."""
        data = self.to_dict()
        unknown = sorted(set(overrides) - set(self.field_names()))
        if unknown:
            raise ConfigError("unknown configuration key(s): " + ", ".join(unknown))
        data.update(overrides)
        return Config.from_dict(data)

    def save(self, path: "str | Path") -> Path:
        path = Path(path)
        if path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


DEFAULT_CONFIG = Config()
