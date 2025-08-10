# Contributing to RL Bot

Thank you for your interest in contributing to the RL Bot project! This document provides guidelines for contributing to this reinforcement learning project.

## Table of Contents
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Install dependencies: `pip install -r requirements.txt`
4. Run the project: `python main.py`

## Development Setup

### Prerequisites
- Python 3.7+
- pygame
- numpy

### Local Development
```bash
# Clone your fork
git clone https://github.com/yourusername/rl-bot.git
cd rl-bot

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
```

## How to Contribute

### Areas for Contribution

1. **Algorithm Improvements**
   - Implement different RL algorithms (DQN, A3C, PPO)
   - Optimize Q-learning parameters
   - Add multi-agent coordination

2. **Environment Enhancements**
   - Add obstacles to the grid
   - Implement different grid sizes
   - Create new reward structures

3. **Visualization Features**
   - Add performance graphs
   - Implement heatmaps for Q-values
   - Create agent path analysis

4. **Performance Optimization**
   - Speed up training
   - Memory optimization
   - Parallel training

5. **Documentation**
   - Improve code comments
   - Add tutorials
   - Create examples

### Types of Contributions

- **Bug fixes**: Fix issues in the existing code
- **Feature additions**: Add new functionality
- **Documentation**: Improve or add documentation
- **Testing**: Add test cases
- **Performance**: Optimize existing code

## Code Style

### Python Style Guidelines
- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Use type hints where appropriate

### Example Code Style
```python
def calculate_reward(new_pos: tuple, visited: set, goal_pos: tuple) -> float:
    """
    Calculate reward based on position and visit history.
    
    Args:
        new_pos: New position coordinates (x, y)
        visited: Set of previously visited positions
        goal_pos: Goal position coordinates (x, y)
    
    Returns:
        float: Calculated reward value
    """
    reward = -1.0  # Step penalty
    
    if new_pos == goal_pos:
        reward = 100.0  # Goal reward
    elif new_pos in visited:
        reward -= 5.0  # Revisit penalty
    
    return reward
```

### Configuration Guidelines
- Keep all configuration parameters in the CONFIG section
- Use descriptive names for parameters
- Add comments explaining parameter effects
- Provide reasonable default values

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, concise commit messages
   - Test your changes thoroughly
   - Update documentation if needed

3. **Submit a pull request**
   - Provide a clear description of changes
   - Reference any related issues
   - Include screenshots for visual changes

### Commit Message Format
```
type(scope): short description

Longer explanation if needed

Fixes #issue-number
```

Examples:
- `feat(agent): add multi-agent coordination`
- `fix(reward): correct reward calculation bug`
- `docs(readme): update installation instructions`

## Reporting Issues

### Bug Reports
When reporting bugs, please include:
- Operating system and Python version
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages or screenshots
- Relevant configuration settings

### Feature Requests
For feature requests, please include:
- Clear description of the feature
- Use case or motivation
- Possible implementation approach
- Any relevant examples or references

### Issue Template
```markdown
## Bug Report / Feature Request

**Type**: Bug / Feature

**Description**: 
Brief description of the issue or feature

**Steps to Reproduce** (for bugs):
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**:
What should happen

**Actual Behavior** (for bugs):
What actually happens

**Environment**:
- OS: 
- Python version: 
- pygame version: 
- numpy version: 

**Additional Context**:
Any other relevant information
```

## Recognition

Contributors will be recognized in:
- The README.md file
- Release notes
- The project's contributors list

Thank you for contributing to RL Bot!
