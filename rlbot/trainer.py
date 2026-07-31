"""Training and evaluation loops.

Everything here is headless and deterministic given a seed. Visualisation is
injected as an optional renderer so training can run in CI, in a notebook, or
behind a Pygame window without changing a line of the algorithm.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set

from rlbot.agent import QLearningAgent
from rlbot.config import Config, Position
from rlbot.environment import GridWorld
from rlbot.replay import Episode, ReplayBuffer

# A renderer receives the live board state; returning False asks training to stop.
Renderer = Callable[["Frame"], bool]


@dataclass
class Frame:
    """Snapshot handed to a renderer once per simulation step."""

    positions: Dict[str, Position]
    paths: Dict[str, List[Position]]
    caption: str


@dataclass
class Rollout:
    """Outcome of one agent's single episode or evaluation run."""

    agent_name: str
    path: List[Position]
    reached_goal: bool
    total_reward: float
    reason: str = "goal"

    @property
    def steps(self) -> int:
        return max(0, len(self.path) - 1)

    def to_dict(self) -> Dict[str, object]:
        return {
            "agent": self.agent_name,
            "steps": self.steps,
            "reached_goal": self.reached_goal,
            "total_reward": round(self.total_reward, 3),
            "reason": self.reason,
            "path": [list(p) for p in self.path],
        }


@dataclass
class EpisodeStats:
    """Per-episode record kept for the learning curve."""

    episode: int
    epsilon: float
    rollouts: Dict[str, Rollout]
    replay_updates: int = 0

    def steps(self, agent_name: str) -> int:
        return self.rollouts[agent_name].steps

    def to_dict(self) -> Dict[str, object]:
        return {
            "episode": self.episode,
            "epsilon": round(self.epsilon, 5),
            "replay_updates": self.replay_updates,
            "agents": {name: r.to_dict() for name, r in self.rollouts.items()},
        }


@dataclass
class TrainingResult:
    """Everything a caller needs after :func:`train` returns."""

    config: Config
    env: GridWorld
    agents: Dict[str, QLearningAgent]
    history: List[EpisodeStats] = field(default_factory=list)
    best: Dict[str, Rollout] = field(default_factory=dict)
    interrupted: bool = False

    @property
    def agent_names(self) -> List[str]:
        return list(self.agents)

    def success_rate(self, agent_name: str) -> float:
        if not self.history:
            return 0.0
        wins = sum(1 for ep in self.history if ep.rollouts[agent_name].reached_goal)
        return wins / len(self.history)

    def steps_series(self, agent_name: str) -> List[int]:
        return [ep.rollouts[agent_name].steps for ep in self.history]

    def optimal_steps(self, agent_name: str) -> Optional[int]:
        start = self.history[0].rollouts[agent_name].path[0] if self.history else None
        if start is None:
            return None
        return self.env.optimal_steps(start)

    def summary(self) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for name in self.agents:
            best = self.best.get(name)
            out[name] = {
                "episodes": len(self.history),
                "success_rate": round(self.success_rate(name), 4),
                "best_steps": best.steps if best else None,
                "optimal_steps": self.optimal_steps(name),
                "states_learned": len(self.agents[name].q),
            }
        return out

    def save_history_csv(self, path: "str | Path") -> Path:
        """Write the learning curve as CSV (one row per episode)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = self.agent_names
        header = ["episode", "epsilon", "replay_updates"]
        for name in names:
            header += [f"{name}_steps", f"{name}_reward", f"{name}_reached_goal"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for ep in self.history:
                row: List[object] = [ep.episode, round(ep.epsilon, 5), ep.replay_updates]
                for name in names:
                    rollout = ep.rollouts[name]
                    row += [
                        rollout.steps,
                        round(rollout.total_reward, 3),
                        int(rollout.reached_goal),
                    ]
                writer.writerow(row)
        return path

    def save_summary_json(self, path: "str | Path") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "summary": self.summary(),
            "best": {name: r.to_dict() for name, r in self.best.items()},
            "interrupted": self.interrupted,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


class _AgentRun:
    """Mutable per-episode bookkeeping for one agent."""

    def __init__(self, agent: QLearningAgent, start: Position, env: GridWorld) -> None:
        self.agent = agent
        self.env = env
        self.position = start
        self.visited: Set[Position] = {start}
        self.path: List[Position] = [start]
        self.episode = Episode(agent.name)
        self.done = False
        self.reason = "max_steps"
        self.last_distance = env.manhattan_distance(start, env.goal)
        self.no_progress = 0

    def take_step(self, epsilon: float) -> None:
        cfg = self.agent.config
        state = self.position
        action = self.agent.select_action(state, self.visited, epsilon=epsilon)
        next_state = self.env.step(state, action)
        reward = self.env.reward(next_state, self.visited)
        reached_goal = self.env.is_terminal(next_state)

        # Detect an agent circling without closing on the goal and cut it loose,
        # otherwise a single bad episode burns the whole max_steps budget.
        distance = self.env.manhattan_distance(next_state, self.env.goal)
        self.no_progress = 0 if distance < self.last_distance else self.no_progress + 1
        self.last_distance = distance

        stuck = (
            not reached_goal
            and self.no_progress > cfg.no_progress_threshold
            and len(self.path) > cfg.max_steps // 2
        )
        if stuck:
            reward += cfg.loop_penalty

        done = reached_goal or stuck
        self.agent.update(state, action, reward, next_state, reached_goal)
        self.episode.add(state, action, reward, next_state, reached_goal)

        self.position = next_state
        self.visited.add(next_state)
        self.path.append(next_state)
        if done:
            self.done = True
            self.reason = "goal" if reached_goal else "stuck"
            self.episode.reached_goal = reached_goal

    def to_rollout(self) -> Rollout:
        return Rollout(
            agent_name=self.agent.name,
            path=list(self.path),
            reached_goal=self.env.is_terminal(self.position),
            total_reward=self.episode.total_reward,
            reason=self.reason,
        )


def build_agents(
    env: GridWorld, config: Config, rng: random.Random
) -> Dict[str, QLearningAgent]:
    """Create the red and blue agents that share ``env``."""
    return {
        "red": QLearningAgent("red", env, config, rng=random.Random(rng.random())),
        "blue": QLearningAgent("blue", env, config, rng=random.Random(rng.random())),
    }


def starts_for(config: Config) -> Dict[str, Position]:
    return {"red": config.red_start, "blue": config.blue_start}


def train(
    config: Optional[Config] = None,
    env: Optional[GridWorld] = None,
    agents: Optional[Dict[str, QLearningAgent]] = None,
    renderer: Optional[Renderer] = None,
    on_episode_end: Optional[Callable[[EpisodeStats], None]] = None,
) -> TrainingResult:
    """Run Q-learning for ``config.episodes`` episodes.

    ``renderer`` is called once per simulation step; returning ``False`` from it
    stops training early and marks the result as interrupted.
    """
    config = config or Config()
    config.validate()
    env = env or GridWorld(config)
    # The env supplies the dynamics while ``config`` supplies the rewards and
    # hyperparameters. If a caller passes an env built from a different config
    # the two disagree silently and the run optimises a board it is not on.
    if env.config is not config:
        board = ("grid_size", "goal", "obstacles")
        mismatched = [
            key
            for key in board
            if getattr(env.config, key) != getattr(config, key)
        ]
        if mismatched:
            raise ValueError(
                "env was built from a different board than the training config; "
                "mismatched: " + ", ".join(mismatched)
            )
    starts = starts_for(config)

    unreachable = [name for name, s in starts.items() if env.optimal_steps(s) is None]
    if unreachable:
        raise ValueError(
            "goal is unreachable from the start position of: " + ", ".join(sorted(unreachable))
        )

    rng = random.Random(config.seed)
    agents = agents or build_agents(env, config, rng)
    missing = sorted(set(agents) - set(starts))
    if missing:
        raise ValueError(
            "no start position for agent(s): " + ", ".join(missing) +
            f"; known agents are {', '.join(sorted(starts))}"
        )
    buffers = {
        name: ReplayBuffer(config.replay_buffer_size, rng=random.Random(rng.random()))
        for name in agents
    }

    result = TrainingResult(config=config, env=env, agents=agents)
    epsilon = config.epsilon_start

    for ep in range(1, config.episodes + 1):
        runs = {name: _AgentRun(agent, starts[name], env) for name, agent in agents.items()}
        step_count = 0

        while step_count < config.max_steps and not all(r.done for r in runs.values()):
            for run in runs.values():
                if not run.done:
                    run.take_step(epsilon)
            step_count += 1

            if renderer is not None:
                progress = "  ".join(
                    f"{name}:{len(run.path) - 1}" for name, run in runs.items()
                )
                caption = (
                    f"Train {ep}/{config.episodes}  {progress}  eps {epsilon:.3f}"
                )
                frame = Frame(
                    positions={n: r.position for n, r in runs.items()},
                    paths={n: list(r.path) for n, r in runs.items()},
                    caption=caption,
                )
                if renderer(frame) is False:
                    result.interrupted = True
                    break

        if result.interrupted:
            break

        rollouts = {name: run.to_rollout() for name, run in runs.items()}
        replay_updates = 0

        for name, run in runs.items():
            rollout = rollouts[name]
            if rollout.reached_goal and rollout.steps <= config.efficient_solution_threshold:
                buffers[name].add_episode(run.episode)
            best = result.best.get(name)
            if rollout.reached_goal and (best is None or rollout.steps < best.steps):
                result.best[name] = rollout

        if ep % config.replay_frequency == 0:
            for name, agent in agents.items():
                replay_updates += buffers[name].replay(
                    agent,
                    sample_size=config.replay_sample_size,
                    alpha_scale=config.replay_alpha_scale,
                )

        stats = EpisodeStats(
            episode=ep, epsilon=epsilon, rollouts=rollouts, replay_updates=replay_updates
        )
        result.history.append(stats)
        if on_episode_end is not None:
            on_episode_end(stats)

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

    return result


def rollout_greedy(
    agent: QLearningAgent,
    env: Optional[GridWorld] = None,
    start: Optional[Position] = None,
    max_steps: Optional[int] = None,
    avoid_visited: bool = True,
) -> Rollout:
    """Follow the greedy policy from ``start`` and report what happened.

    ``avoid_visited`` keeps the walk from spinning in a two-cell cycle when the
    Q-table has not fully converged; it is the same tie-break the agent used
    while exploring, so the reported path is one the policy can really produce.
    """
    env = env or agent.env
    cfg = agent.config
    start = start if start is not None else cfg.red_start
    limit = max_steps if max_steps is not None else cfg.max_test_steps

    position = start
    visited: Set[Position] = {start}
    path: List[Position] = [start]
    total_reward = 0.0
    reason = "max_steps"

    for _ in range(limit):
        if env.is_terminal(position):
            reason = "goal"
            break
        action = agent.select_action(
            position, visited if avoid_visited else None, greedy=True
        )
        next_position = env.step(position, action)
        if next_position == position:
            # Bumping a wall is not a step, so it must not be charged the step
            # and revisit penalties -- doing so made a stuck rollout's reported
            # total_reward depend on which wall it happened to face.
            reason = "stuck"
            break
        total_reward += env.reward(next_position, visited)
        position = next_position
        path.append(position)
        if position in visited and not avoid_visited:
            reason = "loop"
            break
        visited.add(position)
    if env.is_terminal(position):
        reason = "goal"

    return Rollout(
        agent_name=agent.name,
        path=path,
        reached_goal=env.is_terminal(position),
        total_reward=total_reward,
        reason=reason,
    )


def validate_policies(
    result: TrainingResult, verbose: bool = True
) -> Dict[str, Rollout]:
    """Greedy-rollout every trained agent and optionally print the verdict."""
    outcomes: Dict[str, Rollout] = {}
    starts = starts_for(result.config)
    for name, agent in result.agents.items():
        rollout = rollout_greedy(agent, result.env, starts[name])
        outcomes[name] = rollout
        if verbose:
            optimal = result.env.optimal_steps(starts[name])
            verdict = "SUCCESS" if rollout.reached_goal else "FAILED"
            suffix = f" (optimal {optimal})" if optimal is not None else ""
            print(f"  {name:<5} {verdict} in {rollout.steps} steps{suffix}")
    return outcomes


def format_summary(result: TrainingResult) -> str:
    """Multi-line report suitable for stdout."""
    lines: List[str] = []
    for name, stats in result.summary().items():
        lines.append(
            f"  {name:<5} best={stats['best_steps']} optimal={stats['optimal_steps']} "
            f"success_rate={stats['success_rate']:.2f} states={stats['states_learned']}"
        )
    return "\n".join(lines)


def moving_average(values: Sequence[float], window: int = 10) -> List[float]:
    """Simple trailing moving average, used for learning-curve reporting."""
    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    out: List[float] = []
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= window:
            running -= values[i - window]
        out.append(running / min(i + 1, window))
    return out
