# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
