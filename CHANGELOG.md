# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-31

### Added
- Importable `rlbot` package: `config`, `environment`, `agent`, `replay`, `trainer`, `visualizer`, `cli`
- Headless training, so runs no longer require a display or Pygame
- `train`, `replay` and `show` subcommands with hyperparameter and config-file flags
- Obstacle support, arbitrary grid sizes and per-agent start positions
- Breadth-first shortest path used to reject unsolvable boards and to score policies against optimal
- Q-table persistence to JSON, plus `history.csv` and `summary.json` run artifacts
- Seeded, reproducible training
- ASCII renderer for terminal-only environments
- Pytest suite of 94 deterministic tests requiring no dataset or display
- `configs/default.json` and `configs/maze.json` examples
- `pyproject.toml` with a `rlbot` console entry point

### Fixed
- Terminal transitions no longer bootstrap from the goal's own Q-values, which had inflated every Q-value on the board
- Replay buffer now replays transitions in reverse and stores real `done` flags, so goal reward reaches the start of a trajectory
- Greedy validation no longer walks into two-cell cycles and reports honestly when it gets stuck
- Moving into a wall no longer counts as a move, so agents cannot burn steps against the edge
- `config.py` values are actually used; the duplicate hardcoded constants in `main.py` are gone
- Pygame is no longer initialised at import time, so importing the project cannot open a window

### Changed
- `main.py` is a thin entry point; all logic moved into the package
- Config is a validated dataclass; malformed or inconsistent settings fail with a clear message instead of a crash mid-run

## [1.0.0] - 2025-08-10

### Added
- Initial implementation of two-agent Q-learning navigation
- Red and Blue agents with independent Q-tables
- Experience replay buffer system for improved learning
- Real-time visualization using Pygame
- Epsilon-greedy exploration strategy with decay
- Early termination for agents stuck in loops
- Progress validation after training
- Manhattan distance-based progress tracking
- Comprehensive reward system with penalties
- Greedy policy replay demonstration

### Features
- 6x6 grid environment
- 4-directional movement actions
- Configurable hyperparameters
- Visual training progress display
- Performance metrics tracking
- Replay buffer statistics

### Technical Details
- Q-learning algorithm implementation
- Experience replay with best episode selection
- Anti-loop mechanisms
- Policy validation system
- Responsive Pygame interface
