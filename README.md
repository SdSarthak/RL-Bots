# RL Bot — Two-Agent Q-Learning Navigation

Two independent tabular Q-learning agents (**red** and **blue**) learn to cross a
grid world to a shared goal, around obstacles, and their learned policies are
checked against the true shortest path found by breadth-first search.

Training runs headless by default, so it works over SSH and in CI; the Pygame
window is an optional view onto the same loop.

```
$ python main.py --headless --quiet

Training summary:
  red   best=10 optimal=10 success_rate=1.00 states=35
  blue  best=10 optimal=10 success_rate=1.00 states=35
  red   steps moving avg: first10=39.8 last=10.0
  blue  steps moving avg: first10=35.2 last=10.6

Validating greedy policies:
  red   SUCCESS in 10 steps (optimal 10)
  blue  SUCCESS in 10 steps (optimal 10)
```

## Install

```bash
git clone https://github.com/SdSarthak/RL-Bots.git
cd RL-Bots
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Python 3.8 or newer. There is **no dataset to download** — the environment is
generated from the config, so a fresh clone trains in a couple of seconds.

`pygame` is only needed for the window. If it is missing or no display is
available, the CLI says so and continues headless.

## Usage

```bash
python main.py                                  # train with the window, then replay
python main.py --headless --quiet               # train silently, print the summary
python -m rlbot train --episodes 300 --seed 7   # same thing, module form
python -m rlbot show --config configs/maze.json # print the board and optimal distances
```

### Commands

| Command | What it does |
| --- | --- |
| `train` | Trains both agents, validates the greedy policies, optionally saves artifacts, then animates the learned paths. Default command, so the subcommand may be omitted. |
| `replay` | Loads Q-tables saved by `train --save-dir` and walks them greedily. |
| `show` | Prints the board as ASCII with each agent's optimal distance to the goal. |

### Useful flags

| Flag | Meaning |
| --- | --- |
| `--config PATH` | JSON config file (see `configs/`). |
| `--headless` | Never open a window. |
| `--render pygame\|ascii\|none` | Pick the renderer explicitly. |
| `--grid-size N` | Resize the board; the default goal follows the bottom-right corner. |
| `--goal ROW COL` | Place the goal yourself. |
| `--episodes`, `--max-steps` | Training budget. |
| `--alpha`, `--gamma`, `--epsilon-start`, `--epsilon-decay`, `--epsilon-min` | Q-learning hyperparameters. |
| `--seed N` | Seed for reproducible runs. |
| `--save-dir DIR` | Write `red.json`, `blue.json`, `history.csv`, `summary.json` and `config.json`. |
| `--quiet` | Suppress the per-episode log. |

Exit code is `0` when both greedy policies reach the goal, `1` when one does
not, and `2` on a bad or unsolvable configuration.

### Saving and replaying a run

```bash
python -m rlbot train --config configs/maze.json --headless --save-dir runs/maze
python -m rlbot replay --load-dir runs/maze
```

`replay` picks up `runs/maze/config.json` automatically, so the board matches the
one the agents were trained on. `runs/` is gitignored.

## Configuration

Every tunable lives on `rlbot.config.Config` and can be supplied as JSON. Keys
you leave out fall back to the defaults, and unknown keys are rejected rather
than silently ignored.

```json
{
  "grid_size": 8,
  "red_start": [0, 0],
  "blue_start": [7, 0],
  "goal": [7, 7],
  "obstacles": [[1, 1], [1, 2], [3, 4]],
  "episodes": 400,
  "alpha": 0.3,
  "gamma": 0.97,
  "epsilon_decay": 0.985,
  "seed": 7
}
```

Two examples ship with the repo:

- `configs/default.json` — the open 6x6 board, both agents from `(0, 0)` to `(5, 5)`.
- `configs/maze.json` — an 8x8 serpentine maze with dead ends and separate starts.

## How it works

**Environment** (`rlbot/environment.py`) — a square grid. Moves off the edge are
clamped and moves into an obstacle are refused, so the state space stays closed
and the Q-table stays keyed by position. A BFS gives the true shortest path,
which is used both to reject unsolvable configs before training and to report
how close a learned policy came to optimal.

**Rewards** — `+100` for the goal, `-1` per step, an extra `-5` for stepping back
onto a visited cell, and `-10` when an agent is cut loose for circling.

**Agent** (`rlbot/agent.py`) — epsilon-greedy over the actions that actually move
and prefer unvisited cells, falling back to any legal move so a boxed-in agent
never deadlocks. Terminal transitions do not bootstrap: the goal has no
successor, so folding `gamma * max Q(goal)` back in would inflate every value on
the board without bound.

**Replay** (`rlbot/replay.py`) — the shortest successful episodes are kept and
replayed in reverse, which carries the terminal reward all the way to the start
of the trajectory in a single pass. That is what makes 80 episodes enough on the
6x6 board.

**Trainer** (`rlbot/trainer.py`) — runs both agents in lockstep, decays epsilon
per episode, replays every `replay_frequency` episodes and records a full
learning curve. The renderer is injected, so the algorithm is identical headless
and windowed.

### Package layout

```
rlbot/
  config.py        validated dataclass config, JSON load/save
  environment.py   grid dynamics, rewards, BFS shortest path
  agent.py         tabular Q-learning agent, Q-table persistence
  replay.py        episode replay buffer
  trainer.py       training loop, greedy rollouts, metrics
  visualizer.py    Pygame and ASCII renderers
  cli.py           argument parsing and commands
configs/           example JSON configurations
tests/             pytest suite (no dataset, no display)
```

## Controls

While the Pygame window is open, `Esc`, `Q` or closing the window stops training
early and the run still reports what it learned.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

94 tests, all deterministic and offline. They cover grid dynamics and obstacle
handling, terminal-aware Q-updates, replay eviction and back-propagation,
run-to-run reproducibility under a fixed seed, convergence to the BFS-optimal
path on both an open board and a maze, artifact writing, and CLI exit codes.

## Using it as a library

```python
from rlbot import Config, train
from rlbot.trainer import rollout_greedy, starts_for

config = Config(grid_size=8, goal=(7, 7), obstacles=[(1, 1), (1, 2)], episodes=300)
result = train(config)

print(result.summary())
path = rollout_greedy(result.agents["red"], result.env, starts_for(config)["red"])
print(path.steps, "steps, optimal is", result.env.optimal_steps((0, 0)))
```

## Roadmap

- Agent-to-agent coordination instead of two independent learners
- Deep Q-Networks for grids too large to tabulate
- Stochastic transitions and moving goals
- Q-value heatmap overlay in the renderer

## License

MIT — see [LICENSE](LICENSE).
