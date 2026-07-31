"""Tabular Q-learning agent."""

from __future__ import annotations

import json
import math
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
        """Highest-valued allowed action, ties broken with the agent's rng.

        Ties are compared with a tolerance. Two routes of genuinely equal value
        can end up differing by one ULP after a few hundred backups, and an
        exact ``==`` would silently hand every such tie to the lowest action
        index instead of the rng, biasing the learned policy towards "Right".
        """
        candidates = list(range(len(ACTIONS))) if allowed is None else list(allowed)
        if not candidates:
            raise ValueError("best_action needs at least one allowed action")
        for action in candidates:
            if not 0 <= action < len(ACTIONS):
                raise IndexError(
                    f"action {action} out of range (0..{len(ACTIONS) - 1})"
                )
        values = self.q_values(state)
        chosen = [float(values[a]) for a in candidates]
        if not all(math.isfinite(v) for v in chosen):
            # NaN never compares equal to itself, so the tie-break below would
            # come up empty and rng.choice would raise IndexError several
            # frames away from the real cause.
            raise ValueError(
                f"Q-values for state {state} are not finite ({chosen}); the "
                "run diverged or the loaded Q-table is corrupt"
            )
        best = max(chosen)
        tied = [
            action
            for action, value in zip(candidates, chosen)
            if math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-12)
        ]
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
        """Human-readable greedy policy over the states seen so far.

        Restricted to :meth:`allowed_actions`, exactly like :meth:`select_action`.
        A plain ``argmax`` over the raw row reports moves the agent can never
        make -- walking into a wall or an obstacle -- because those actions keep
        their zero initialisation while every real move is driven negative by
        the step penalty. On the maze board that made most of the printed
        policy disagree with the path the agent actually walks.
        """
        out: Dict[Position, str] = {}
        for state in sorted(self.q):
            values = self.q[state]
            allowed = self.allowed_actions(state)
            best = max(allowed, key=lambda a: float(values[a]))
            out[state] = ACTION_NAMES[best]
        return out

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
        """Replace the Q-table from a :meth:`state_dict` payload.

        The table is checked against this agent's environment. A Q-table
        trained on an 8x8 maze loaded onto a 6x6 board used to be accepted in
        silence: the out-of-grid rows were simply never reached, so the agent
        wandered on zero-initialised values while the CLI still reported a
        greedy rollout as if it were the trained policy.
        """
        table = data.get("q")
        if not isinstance(table, dict):
            raise ValueError("Q-table payload must contain a 'q' object")
        size = self.env.size
        loaded: Dict[Position, np.ndarray] = {}
        for key, values in table.items():
            try:
                row_s, col_s = str(key).split(",")
                state = (int(row_s), int(col_s))
            except ValueError as exc:
                raise ValueError(f"malformed Q-table state key {key!r}") from exc
            if not self.env.in_bounds(state):
                raise ValueError(
                    f"Q-table state {state} is outside the {size}x{size} grid; "
                    "this table was trained on a different board"
                )
            try:
                array = np.asarray(values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"state {state} has non-numeric action values: {values!r}"
                ) from exc
            if array.shape != (len(ACTIONS),):
                raise ValueError(
                    f"state {state} has {array.size} action values, expected {len(ACTIONS)}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"state {state} has non-finite action values: {values!r}")
            loaded[state] = array
        # Only swap the table in once the whole payload has been accepted, so a
        # failed load leaves the agent usable instead of half-overwritten.
        self.q = loaded
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
