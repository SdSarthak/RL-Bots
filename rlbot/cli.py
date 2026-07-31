"""Command line interface for RL Bot.

    python -m rlbot train --episodes 200 --headless
    python -m rlbot train --config configs/maze.json --save-dir runs/maze
    python -m rlbot replay --load-dir runs/maze
    python -m rlbot show --config configs/maze.json
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rlbot.agent import QLearningAgent
from rlbot.config import Config, ConfigError
from rlbot.environment import GridWorld
from rlbot.trainer import (
    EpisodeStats,
    Rollout,
    format_summary,
    moving_average,
    rollout_greedy,
    starts_for,
    train,
    validate_policies,
)

AGENT_NAMES = ("red", "blue")


# ----------------------------------------------------------------------
# argument plumbing
# ----------------------------------------------------------------------
def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="JSON config file to load")
    parser.add_argument("--grid-size", type=int, help="grid width and height")
    parser.add_argument(
        "--goal",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        help="goal cell; defaults to the bottom-right corner of the grid",
    )
    parser.add_argument("--seed", type=int, help="random seed (default 0)")
    parser.add_argument(
        "--render",
        choices=("pygame", "ascii", "none"),
        help="renderer to use (default: pygame, or none when --headless)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="never open a window; equivalent to --render none",
    )
    parser.add_argument("--fps", type=int, help="animation speed for the pygame renderer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlbot",
        description="Two-agent Q-learning navigation on a grid world.",
    )
    subparsers = parser.add_subparsers(dest="command")

    train_p = subparsers.add_parser("train", help="train both agents")
    _add_common_options(train_p)
    train_p.add_argument("--episodes", type=int, help="number of training episodes")
    train_p.add_argument("--max-steps", type=int, help="step cap per episode")
    train_p.add_argument("--alpha", type=float, help="learning rate")
    train_p.add_argument("--gamma", type=float, help="discount factor")
    train_p.add_argument("--epsilon-start", type=float, help="initial exploration rate")
    train_p.add_argument("--epsilon-decay", type=float, help="per-episode epsilon decay")
    train_p.add_argument("--epsilon-min", type=float, help="exploration floor")
    train_p.add_argument(
        "--save-dir",
        type=Path,
        help="directory for learned Q-tables, history.csv and summary.json",
    )
    train_p.add_argument(
        "--no-replay-demo",
        action="store_true",
        help="skip the greedy replay animation after training",
    )
    train_p.add_argument("--quiet", action="store_true", help="only print the final summary")

    replay_p = subparsers.add_parser("replay", help="replay saved Q-tables greedily")
    _add_common_options(replay_p)
    replay_p.add_argument(
        "--load-dir",
        type=Path,
        required=True,
        help="directory holding red.json and blue.json Q-tables",
    )

    show_p = subparsers.add_parser("show", help="print the board and exit")
    _add_common_options(show_p)

    return parser


_CONFIG_OVERRIDES = (
    "grid_size",
    "seed",
    "fps",
    "episodes",
    "max_steps",
    "alpha",
    "gamma",
    "epsilon_start",
    "epsilon_decay",
    "epsilon_min",
)


def _fit_to_grid(config: Config, size: int, explicit_goal: bool) -> Dict[str, object]:
    """Keep the goal and starts inside a resized grid.

    ``--grid-size 4`` on a config whose goal is (5, 5) would otherwise be an
    instant validation error, which is a poor experience for the single most
    obvious flag in the tool.
    """
    fixes: Dict[str, object] = {}
    last = size - 1

    def clamp(pos):
        return (max(0, min(last, pos[0])), max(0, min(last, pos[1])))

    if not explicit_goal and (config.goal[0] > last or config.goal[1] > last):
        fixes["goal"] = (last, last)
    goal = fixes.get("goal", config.goal)

    for key in ("red_start", "blue_start"):
        start = clamp(getattr(config, key))
        if start != getattr(config, key):
            fixes[key] = start
    if any(fixes.get(k, getattr(config, k)) == goal for k in ("red_start", "blue_start")):
        # A clamped start landing on the goal would make the episode trivial.
        fixes.setdefault("red_start", config.red_start)
        for key in ("red_start", "blue_start"):
            if fixes.get(key, getattr(config, key)) == goal:
                fixes[key] = (0, 0) if goal != (0, 0) else (0, 1)
    return fixes


def config_from_args(args: argparse.Namespace) -> Config:
    """Load the config file (if any) and apply CLI overrides on top."""
    config = Config.from_file(args.config) if getattr(args, "config", None) else Config()
    overrides: Dict[str, object] = {
        key: getattr(args, key)
        for key in _CONFIG_OVERRIDES
        if getattr(args, key, None) is not None
    }

    goal = getattr(args, "goal", None)
    if goal is not None:
        overrides["goal"] = (goal[0], goal[1])
    if args.grid_size is not None:
        overrides.update(_fit_to_grid(config, args.grid_size, explicit_goal=goal is not None))

    if getattr(args, "headless", False):
        overrides["show_training"] = False
    if overrides:
        config = config.replace(**overrides)
    return config


def resolve_render_mode(args: argparse.Namespace) -> str:
    if getattr(args, "headless", False):
        return "none"
    return getattr(args, "render", None) or "pygame"


def make_renderer(mode: str, env: GridWorld, config: Config):
    """Build a renderer, degrading to ascii/none when no display is available."""
    if mode == "none":
        return None
    if mode == "ascii":
        from rlbot.visualizer import AsciiRenderer

        return AsciiRenderer(env, every=max(1, config.grid_size))

    from rlbot.visualizer import DisplayUnavailable, PygameRenderer, display_available

    if not display_available():
        print("No display detected; continuing headless.", file=sys.stderr)
        return None
    try:
        return PygameRenderer(env, config).open()
    except DisplayUnavailable as exc:
        print(f"{exc}. Continuing headless.", file=sys.stderr)
        return None


def close_renderer(renderer) -> None:
    if renderer is not None and hasattr(renderer, "close"):
        renderer.close()


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
def _print_board(env: GridWorld, config: Config) -> None:
    print(f"Grid {config.grid_size}x{config.grid_size}, goal {config.goal}")
    print(env.render_ascii({"red": config.red_start, "blue": config.blue_start}))
    for name, start in starts_for(config).items():
        optimal = env.optimal_steps(start)
        reachable = "unreachable" if optimal is None else f"{optimal} steps optimal"
        print(f"  {name:<5} start {start} -> {reachable}")


def cmd_show(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    _print_board(GridWorld(config), config)
    return 0


def _save_artifacts(result, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, agent in result.agents.items():
        agent.save(save_dir / f"{name}.json")
    result.save_history_csv(save_dir / "history.csv")
    result.save_summary_json(save_dir / "summary.json")
    result.config.save(save_dir / "config.json")
    print(f"Saved Q-tables and metrics to {save_dir}")


def cmd_train(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    env = GridWorld(config)

    if not env.is_solvable(starts_for(config).values()):
        print("Goal is unreachable from at least one start position.", file=sys.stderr)
        _print_board(env, config)
        return 2

    quiet = getattr(args, "quiet", False)
    render_mode = resolve_render_mode(args)
    if not config.show_training and render_mode != "none":
        render_mode = "none"
    renderer = make_renderer(render_mode, env, config)

    def report(stats: EpisodeStats) -> None:
        if quiet:
            return
        parts = "  ".join(
            f"{name}={r.steps:>3}{'*' if r.reached_goal else ' '}"
            for name, r in stats.rollouts.items()
        )
        print(f"Episode {stats.episode:>4}/{config.episodes}  {parts}  eps={stats.epsilon:.3f}")

    # The renderer owns an SDL window and an initialised pygame; anything that
    # escapes from here used to leave both alive, which on Windows keeps a dead
    # window on screen until the interpreter exits.
    try:
        try:
            result = train(config, env=env, renderer=renderer, on_episode_end=report)
        except KeyboardInterrupt:
            print("\nInterrupted by user.", file=sys.stderr)
            return 130

        if result.interrupted:
            print("\nTraining window closed; stopping early.", file=sys.stderr)

        print("\nTraining summary:")
        print(format_summary(result))

        for name in result.agents:
            series = result.steps_series(name)
            if len(series) >= 20:
                trend = moving_average([float(s) for s in series], window=10)
                print(f"  {name:<5} steps moving avg: first10={trend[9]:.1f} last={trend[-1]:.1f}")

        print("\nValidating greedy policies:")
        outcomes = validate_policies(result)

        if getattr(args, "save_dir", None):
            try:
                _save_artifacts(result, args.save_dir)
            except OSError as exc:
                print(f"Training finished but saving to {args.save_dir} failed: {exc}",
                      file=sys.stderr)
                return 3

        if renderer is not None and not args.no_replay_demo and not result.interrupted:
            from rlbot.visualizer import replay_paths

            replay_paths(renderer, {n: r.path for n, r in outcomes.items()})
    finally:
        close_renderer(renderer)

    return 0 if all(r.reached_goal for r in outcomes.values()) else 1


def load_agents(load_dir: Path, env: GridWorld, config: Config) -> Dict[str, QLearningAgent]:
    agents: Dict[str, QLearningAgent] = {}
    for name in AGENT_NAMES:
        path = load_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing Q-table {path}")
        agent = QLearningAgent(name, env, config, rng=random.Random(config.seed))
        agent.load(path)
        agents[name] = agent
    return agents


def cmd_replay(args: argparse.Namespace) -> int:
    load_dir: Path = args.load_dir
    saved_config = load_dir / "config.json"
    if getattr(args, "config", None) is None and saved_config.exists():
        args.config = saved_config
    config = config_from_args(args)
    env = GridWorld(config)

    try:
        agents = load_agents(load_dir, env, config)
    except (OSError, ValueError) as exc:
        print(f"Cannot replay: {exc}", file=sys.stderr)
        return 2

    starts = starts_for(config)
    rollouts: Dict[str, Rollout] = {}
    for name, agent in agents.items():
        rollout = rollout_greedy(agent, env, starts[name])
        rollouts[name] = rollout
        optimal = env.optimal_steps(starts[name])
        verdict = "reached goal" if rollout.reached_goal else f"stopped ({rollout.reason})"
        print(f"  {name:<5} {verdict} in {rollout.steps} steps (optimal {optimal})")

    renderer = make_renderer(resolve_render_mode(args), env, config)
    if renderer is not None:
        from rlbot.visualizer import replay_paths

        try:
            replay_paths(renderer, {n: r.path for n, r in rollouts.items()})
        finally:
            close_renderer(renderer)
    else:
        for name, rollout in rollouts.items():
            print(f"\n{name} path:")
            print(env.render_ascii({name: rollout.path[-1]}))

    return 0 if all(r.reached_goal for r in rollouts.values()) else 1


COMMANDS = {"train": cmd_train, "replay": cmd_replay, "show": cmd_show}


def normalise_argv(argv: Optional[Sequence[str]] = None) -> List[str]:
    """Default to the ``train`` subcommand so bare flags still work.

    ``python main.py --headless`` should train headlessly rather than fail on an
    unrecognised top-level option.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return ["train"]
    if args[0] in COMMANDS or args[0] in ("-h", "--help"):
        return args
    return ["train", *args]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(normalise_argv(argv))
    try:
        return COMMANDS[args.command](args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
