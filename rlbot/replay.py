"""Experience replay over whole successful episodes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence

from rlbot.config import Position

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rlbot.agent import QLearningAgent


@dataclass(frozen=True)
class Transition:
    """A single ``(s, a, r, s', done)`` tuple."""

    state: Position
    action: int
    reward: float
    next_state: Position
    done: bool


class Episode:
    """An ordered list of transitions collected during one episode."""

    __slots__ = ("agent_name", "transitions", "total_reward", "reached_goal")

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.transitions: List[Transition] = []
        self.total_reward = 0.0
        self.reached_goal = False

    def add(
        self,
        state: Position,
        action: int,
        reward: float,
        next_state: Position,
        done: bool,
    ) -> None:
        self.transitions.append(Transition(state, action, reward, next_state, done))
        self.total_reward += reward

    def __len__(self) -> int:
        return len(self.transitions)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self.transitions)

    @property
    def steps(self) -> int:
        return len(self.transitions)

    def states(self) -> List[Position]:
        if not self.transitions:
            return []
        return [self.transitions[0].state] + [t.next_state for t in self.transitions]


class ReplayBuffer:
    """Keeps the shortest successful episodes and re-trains on them.

    Replaying whole trajectories, rather than shuffled transitions, propagates
    the terminal reward backwards along a route the agent already knows works,
    which is what makes 80 episodes enough on this board.
    """

    def __init__(self, max_size: int = 8, rng: Optional[random.Random] = None) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self.rng = rng or random.Random()
        self.buffer: List[Episode] = []

    def __len__(self) -> int:
        return len(self.buffer)

    def add_episode(self, episode: Episode) -> bool:
        """Store ``episode``, evicting the longest when the buffer overflows.

        Returns ``True`` when the episode was retained.
        """
        if not episode.transitions:
            return False
        self.buffer.append(episode)
        self.buffer.sort(key=lambda ep: (ep.steps, -ep.total_reward))
        if len(self.buffer) > self.max_size:
            evicted = self.buffer[self.max_size:]
            self.buffer = self.buffer[: self.max_size]
            return episode not in evicted
        return True

    def sample(self, sample_size: int) -> List[Episode]:
        if sample_size < 1 or not self.buffer:
            return []
        return self.rng.sample(self.buffer, min(sample_size, len(self.buffer)))

    def replay(
        self,
        agent: "QLearningAgent",
        sample_size: int = 3,
        alpha_scale: float = 0.5,
        min_episodes: int = 2,
    ) -> int:
        """Re-run sampled episodes through ``agent.update``.

        Updates run in reverse so the goal reward reaches the start of the
        trajectory within a single pass. Returns the number of backups applied.
        """
        if len(self.buffer) < min_episodes:
            return 0
        updates = 0
        for episode in self.sample(sample_size):
            for transition in reversed(episode.transitions):
                agent.update(
                    transition.state,
                    transition.action,
                    transition.reward,
                    transition.next_state,
                    transition.done,
                    alpha_scale=alpha_scale,
                )
                updates += 1
        return updates

    def best_episode(self) -> Optional[Episode]:
        return self.buffer[0] if self.buffer else None

    def episode_lengths(self) -> Sequence[int]:
        return [ep.steps for ep in self.buffer]

    def clear(self) -> None:
        self.buffer.clear()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ReplayBuffer(size={len(self.buffer)}/{self.max_size})"
