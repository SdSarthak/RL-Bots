"""
Configuration file for RL Bot project.
Contains all hyperparameters and settings for the reinforcement learning environment.
"""

# -------- ENVIRONMENT CONFIGURATION --------
GRID_SIZE = 6           # Size of the grid world (GRID_SIZE x GRID_SIZE)
CELL_SIZE = 50          # Size of each cell in pixels for visualization

# -------- TRAINING CONFIGURATION --------
EPISODES = 80           # Number of training episodes
MAX_STEPS = 200         # Maximum steps per episode to prevent infinite loops

# -------- Q-LEARNING HYPERPARAMETERS --------
ALPHA = 0.2             # Learning rate (how much new information overrides old)
GAMMA = 0.95            # Discount factor (importance of future rewards)

# -------- EXPLORATION PARAMETERS --------
EPSILON_START = 1.0     # Initial exploration probability (100% random)
EPSILON_DECAY = 0.92    # Decay rate for exploration (multiplicative)
EPSILON_MIN = 0.02      # Minimum exploration probability (2% random)

# -------- VISUALIZATION SETTINGS --------
FPS = 20                # Frames per second for visualization
SHOW_TRAINING = True    # Whether to show visualization during training (set False for faster training)

# -------- ENVIRONMENT POSITIONS --------
RED_START = (0, 0)      # Starting position for red agent
BLUE_START = (0, 0)     # Starting position for blue agent
GOAL = (5, 5)           # Goal position for both agents

# -------- REPLAY BUFFER SETTINGS --------
REPLAY_BUFFER_SIZE = 8  # Maximum number of episodes to store in replay buffer
REPLAY_FREQUENCY = 10   # How often to perform replay training (every N episodes)
REPLAY_SAMPLE_SIZE = 3  # Number of episodes to sample for replay training

# -------- REWARD SYSTEM --------
STEP_PENALTY = -1.0     # Penalty for each step taken
GOAL_REWARD = 100.0     # Reward for reaching the goal
REVISIT_PENALTY = -5.0  # Additional penalty for revisiting a cell
LOOP_PENALTY = -10.0    # Penalty for getting stuck in loops

# -------- EARLY TERMINATION SETTINGS --------
NO_PROGRESS_THRESHOLD = 10      # Steps without progress before forced termination
EFFICIENT_SOLUTION_THRESHOLD = 50  # Maximum steps to consider a solution "efficient"

# -------- VALIDATION SETTINGS --------
MAX_TEST_STEPS = 100    # Maximum steps for policy validation

# -------- ACTIONS --------
# Movement actions: (dx, dy) where x is row and y is column
ACTIONS = [
    (0, 1),   # Right
    (0, -1),  # Left
    (1, 0),   # Down
    (-1, 0)   # Up
]

# Action names for debugging/logging
ACTION_NAMES = ["Right", "Left", "Down", "Up"]

# -------- COLORS FOR VISUALIZATION --------
COLORS = {
    'WHITE': (255, 255, 255),
    'GRID': (210, 210, 210),
    'RED_AGENT': (200, 0, 0),
    'BLUE_AGENT': (0, 0, 200),
    'RED_PATH': (255, 200, 200),
    'BLUE_PATH': (200, 200, 255),
    'GOAL': (0, 200, 0),
    'BLACK': (0, 0, 0)
}

# -------- PERFORMANCE TRACKING --------
TRACK_BEST_EPISODES = True     # Whether to track best performing episodes
PRINT_EPISODE_STATS = True     # Whether to print statistics for each episode
PRINT_REPLAY_STATS = True      # Whether to print replay buffer statistics
