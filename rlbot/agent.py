"""Tabular Q-learning agent."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import AbstractSet, Dict, List, Optional

import numpy as np

from rlbot.config import ACTION_NAMES, ACTIONS, Config, Position
from rlbot.environment import GridWorld


class QLearningAgent:
    """An epsilon-greedy tabular Q-learner over grid positions.

    The Q-table is a ``dict`` keyed by position, so unvisited cells cost
    nothing and the agent works unchanged on any grid size.
    """

    def __init__(
        self,
        name: str,
        env: GridWorld,
        config: Optional[Config] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.name = name
        self.env = env
        self.config = config or env.config
        self.rng = rng or random.Random()
        self.q: Dict[Position, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Q-table access
    # ------------------------------------------------------------------
    def q_values(self, state: Position) -> np.ndarray:
        """Q-row for ``state``, created lazily and zero-initialised."""
        row = self.q.get(state)
        if row is None:
            row = np.zeros(len(ACTIONS), dtype=float)
            self.q[state] = row
        return row

    def best_action(self, state: Position, allowed: Optional[List[int]] = None) -> int:
        """Highest-valued allowed action, ties broken with the agent's rng."""
        candidates = list(range(len(ACTIONS))) if allowed is None else list(allowed)
        if not candidates:
            raise ValueError("best_action needs at least one allowed action")
        values = self.q_values(state)
        best = max(values[a] for a in candidates)
        tied = [a for a in candidates if values[a] == best]
        return tied[0] if len(tied) == 1 else self.rng.choice(tied)

    def allowed_actions(
        self, state: Position, visited: Optional[AbstractSet[Position]] = None
    ) -> List[int]:
        """Actions that make progress: they move, and avoid ``visited`` cells.

        Falling back to every action when nothing qualifies is what keeps a
        boxed-in agent from deadlocking; it can then step back over its trail.
        """
        all_actions = list(range(len(ACTIONS)))
        if not visited:
            moving = [a for a in all_actions if self.env.step(state, a) != state]
            return moving or all_actions

        preferred = []
        moving = []
        for action in all_actions:
            nxt = self.env.step(state, action)
            if nxt == state:
                continue
            moving.append(action)
            if nxt not in visited:
                preferred.append(action)
        return preferred or moving or all_actions

    # ------------------------------------------------------------------
    # policy
    # ------------------------------------------------------------------
    def select_action(
        self,
        state: Position,
        visited: Optional[AbstractSet[Position]] = None,
        epsilon: float = 0.0,
        greedy: bool = False,
    ) -> int:
        """Epsilon-greedy choice restricted to :meth:`allowed_actions`."""
        allowed = self.allowed_actions(state, visited)
        if not greedy and epsilon > 0.0 and self.rng.random() < epsilon:
            return self.rng.choice(allowed)
        return self.best_action(state, allowed)

    def policy(self) -> Dict[Position, str]:
        """Human-readable greedy policy over the states seen so far."""
        return {
            state: ACTION_NAMES[int(np.argmax(values))]
            for state, values in sorted(self.q.items())
        }

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def update(
        self,
        state: Position,
        action: int,
        reward: float,
        next_state: Position,
        done: bool,
        alpha_scale: float = 1.0,
    ) -> float:
        """One Q-learning backup. Returns the TD error.

        Terminal transitions bootstrap from nothing: the goal has no successor,
        so folding ``gamma * max Q(goal)`` back in would inflate every value on
        the board without bound.
        """
        cfg = self.config
        row = self.q_values(state)
        target = reward if done else reward + cfg.gamma * float(np.max(self.q_values(next_state)))
        td_error = target - row[action]
        row[action] += cfg.alpha * alpha_scale * td_error
        return float(td_error)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "actions": list(ACTION_NAMES),
            "q": {f"{r},{c}": [float(v) for v in values] for (r, c), values in sorted(self.q.items())},
        }

    def load_state_dict(self, data: Dict[str, object]) -> "QLearningAgent":
        table = data.get("q")
        if not isinstance(table, dict):
            raise ValueError("Q-table payload must contain a 'q' object")
        self.q = {}
        for key, values in table.items():
            try:
                row_s, col_s = str(key).split(",")
                state = (int(row_s), int(col_s))
            except ValueError as exc:
                raise ValueError(f"malformed Q-table state key {key!r}") from exc
            array = np.asarray(values, dtype=float)
            if array.shape != (len(ACTIONS),):
                raise ValueError(
                    f"state {state} has {array.size} action values, expected {len(ACTIONS)}"
                )
            self.q[state] = array
        return self

    def save(self, path: "str | Path") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    def load(self, path: "str | Path") -> "QLearningAgent":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise FileNotFoundError(f"cannot read Q-table {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        return self.load_state_dict(data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"QLearningAgent(name={self.name!r}, states_seen={len(self.q)})"
