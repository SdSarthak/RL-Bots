# RL Bot - Two-Agent Q-Learning Navigation

A reinforcement learning project featuring two agents (Red and Blue) learning to navigate a grid environment to reach a goal using Q-learning with experience replay.

## Overview

This project implements a multi-agent reinforcement learning environment where two agents independently learn optimal navigation policies using Q-learning. The agents must navigate from their starting positions to a goal while avoiding revisiting cells and minimizing the number of steps taken.

## Features

- **Two Independent Agents**: Red and Blue agents learning simultaneously
- **Q-Learning Algorithm**: Classic reinforcement learning with epsilon-greedy exploration
- **Experience Replay**: Replay buffer for improved learning from successful episodes
- **Real-time Visualization**: Pygame-based visual representation of the learning process
- **Adaptive Termination**: Early termination for agents stuck in loops
- **Progress Validation**: Policy validation after training completion

## Environment Details

- **Grid Size**: 6x6 grid world
- **Start Position**: Both agents start at (0, 0)
- **Goal Position**: (5, 5)
- **Actions**: 4 directional movements (Up, Down, Left, Right)
- **Rewards**:
  - Goal reached: +100
  - Step penalty: -1
  - Revisiting penalty: -5
  - Loop penalty: -10

## Algorithm Components

### Q-Learning Parameters
- **Learning Rate (α)**: 0.2
- **Discount Factor (γ)**: 0.95
- **Exploration Rate (ε)**: Starts at 1.0, decays to 0.02
- **Epsilon Decay**: 0.92 per episode

### Experience Replay
- Buffer size: 8 successful episodes per agent
- Replay frequency: Every 10 episodes
- Sample size: 3 episodes for additional training

## Installation

### Prerequisites
```bash
pip install pygame numpy
```

### Running the Project
```bash
python main.py
```

## Configuration

You can modify the following parameters in the `main.py` file:

```python
GRID_SIZE = 6          # Size of the grid environment
CELL_SIZE = 50         # Visual cell size in pixels
FPS = 20               # Animation speed
EPISODES = 80          # Number of training episodes
ALPHA = 0.2            # Learning rate
GAMMA = 0.95           # Discount factor
EPSILON_START = 1.0    # Initial exploration rate
EPSILON_DECAY = 0.92   # Exploration decay rate
EPSILON_MIN = 0.02     # Minimum exploration rate
MAX_STEPS = 200        # Maximum steps per episode
SHOW_TRAINING = True   # Show visualization during training
```

## How It Works

1. **Training Phase**: 
   - Agents explore the environment using epsilon-greedy policy
   - Q-values are updated using the Q-learning algorithm
   - Successful episodes are stored in replay buffers
   - Periodic replay training improves convergence

2. **Validation Phase**:
   - Greedy policies are tested to ensure goal reachability
   - Validation results are displayed

3. **Replay Phase**:
   - Final demonstration using purely greedy policies
   - Shows the learned optimal (or near-optimal) paths

## Visual Interface

- **Red Agent**: Red square representing the first agent
- **Blue Agent**: Blue square representing the second agent
- **Goal**: Green square at position (5, 5)
- **Visited Paths**: Light colored trails showing agent movement history
- **Grid**: Gray grid lines for position reference

## Performance Metrics

The system tracks:
- Steps taken per episode for each agent
- Best performance achieved
- Replay buffer statistics
- Final validation results

## Future Enhancements

- Multi-agent coordination and communication
- Different reward structures
- Larger grid environments
- Obstacle avoidance
- Deep Q-Networks (DQN) implementation
- Comparative analysis with other RL algorithms

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to contribute by:
- Reporting bugs
- Suggesting new features
- Improving the algorithm
- Adding new visualization features
- Optimizing performance

## Author

Created as a reinforcement learning exploration project demonstrating multi-agent Q-learning with experience replay.
