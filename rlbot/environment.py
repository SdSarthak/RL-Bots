"""Grid world the agents navigate.

The environment is deliberately stateless: every method takes the caller's
position explicitly. That keeps two agents on one board trivial to run and
makes the dynamics easy to test without a training loop.
"""

from __future__ import annotations

from collections import deque
from typing import AbstractSet, Dict, Iterable, List, Optional, Tuple

from rlbot.config import ACTIONS, Config, Position


class GridWorld:
    """A square grid with clamped walls, optional obstacles and a single goal."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.size = self.config.grid_size
        self.goal: Position = self.config.goal
        self.obstacles: frozenset = frozenset(self.config.obstacles)
        self.n_actions = len(ACTIONS)

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def in_bounds(self, pos: Position) -> bool:
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size

    def is_blocked(self, pos: Position) -> bool:
        return pos in self.obstacles

    def is_walkable(self, pos: Position) -> bool:
        return self.in_bounds(pos) and not self.is_blocked(pos)

    def step(self, pos: Position, action: int) -> Position:
        """Apply ``action`` to ``pos``.

        Moves that leave the grid are clamped and moves into an obstacle are
        refused, so the agent simply stays put. Both cases keep the state space
        closed, which matters because the Q-table is keyed by position.
        """
        if not 0 <= action < self.n_actions:
            raise IndexError(f"action {action} out of range (0..{self.n_actions - 1})")
        d_row, d_col = ACTIONS[action]
        new_row = max(0, min(self.size - 1, pos[0] + d_row))
        new_col = max(0, min(self.size - 1, pos[1] + d_col))
        candidate = (new_row, new_col)
        if self.is_blocked(candidate):
            return pos
        return candidate

    def neighbours(self, pos: Position) -> List[Position]:
        """Distinct reachable cells one action away from ``pos``."""
        seen: List[Position] = []
        for action in range(self.n_actions):
            nxt = self.step(pos, action)
            if nxt != pos and nxt not in seen:
                seen.append(nxt)
        return seen

    def walkable_cells(self) -> List[Position]:
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in self.obstacles
        ]

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def is_terminal(self, pos: Position) -> bool:
        return pos == self.goal

    def reward(self, new_pos: Position, visited: AbstractSet[Position]) -> float:
        """Reward for landing on ``new_pos`` given the cells already seen."""
        cfg = self.config
        if new_pos == self.goal:
            return cfg.goal_reward
        reward = cfg.step_penalty
        if new_pos in visited:
            reward += cfg.revisit_penalty
        return reward

    @staticmethod
    def manhattan_distance(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------------------------------------------------------------
    # analysis helpers
    # ------------------------------------------------------------------
    def shortest_path(self, start: Position, goal: Optional[Position] = None) -> Optional[List[Position]]:
        """Breadth-first shortest path, or ``None`` when the goal is walled off.

        Used to report how close a learned policy came to optimal and to fail
        fast on configs whose goal is unreachable.
        """
        goal = self.goal if goal is None else goal
        if not self.is_walkable(start) or not self.is_walkable(goal):
            return None
        if start == goal:
            return [start]

        previous: Dict[Position, Optional[Position]] = {start: None}
        queue: deque = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in self.neighbours(current):
                if nxt in previous:
                    continue
                previous[nxt] = current
                if nxt == goal:
                    path = [nxt]
                    node: Optional[Position] = current
                    while node is not None:
                        path.append(node)
                        node = previous[node]
                    path.reverse()
                    return path
                queue.append(nxt)
        return None

    def optimal_steps(self, start: Position, goal: Optional[Position] = None) -> Optional[int]:
        """Number of moves on the shortest path, or ``None`` if unreachable."""
        path = self.shortest_path(start, goal)
        return None if path is None else len(path) - 1

    def is_solvable(self, starts: Iterable[Position]) -> bool:
        return all(self.optimal_steps(start) is not None for start in starts)

    def render_ascii(self, agents: Optional[Dict[str, Position]] = None) -> str:
        """Plain-text board, handy for logs, tests and headless runs."""
        agents = agents or {}
        marks = {pos: name[0].upper() for name, pos in agents.items()}
        rows = []
        for r in range(self.size):
            cells = []
            for c in range(self.size):
                pos = (r, c)
                if pos in marks:
                    cells.append(marks[pos])
                elif pos in self.obstacles:
                    cells.append("#")
                elif pos == self.goal:
                    cells.append("G")
                else:
                    cells.append(".")
            rows.append(" ".join(cells))
        return "\n".join(rows)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"GridWorld(size={self.size}, goal={self.goal}, "
            f"obstacles={len(self.obstacles)})"
        )
