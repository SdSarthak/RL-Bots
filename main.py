# two_agents_qlearn.py
import pygame
import random
import numpy as np
import time

# -------- CONFIG --------
GRID_SIZE = 6
CELL_SIZE = 50
FPS = 20                # Visual speed (higher => faster animation)
EPISODES = 80
ALPHA = 0.2
GAMMA = 0.95
EPSILON_START = 1.0     # start exploration prob
EPSILON_DECAY = 0.92    # faster decay (was 0.995)
EPSILON_MIN = 0.02
MAX_STEPS = 200         # per episode cap to avoid runaway
SHOW_TRAINING = True    # set False to suppress drawing during training (faster)

# Points
RED_START = (0, 0)
BLUE_START = (0, 0)
GOAL = (5, 5)

# Actions: (dx, dy)
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

# -------- Q-tables --------
q_red = {}    # dict mapping state -> np.array(len(ACTIONS))
q_blue = {}

# -------- Replay Buffer --------
class ReplayBuffer:
    def __init__(self, max_size=10):
        self.max_size = max_size
        self.buffer = []
    
    def add_episode(self, agent_name, episode_data):
        """Add successful episode data: (states, actions, rewards)"""
        self.buffer.append((agent_name, episode_data))
        # Keep only the best episodes (shortest successful runs)
        if len(self.buffer) > self.max_size:
            # Sort by episode length (number of steps) and keep the best
            self.buffer.sort(key=lambda x: len(x[1]['states']))
            self.buffer = self.buffer[:self.max_size]
    
    def sample_and_train(self, q_table, sample_size=3):
        """Sample from buffer and perform additional Q-learning updates"""
        if len(self.buffer) < 2:
            return
        
        # Sample episodes from buffer
        episodes_to_replay = random.sample(self.buffer, min(sample_size, len(self.buffer)))
        
        for agent_name, episode_data in episodes_to_replay:
            states = episode_data['states']
            actions = episode_data['actions']
            rewards = episode_data['rewards']
            
            # Replay the episode with Q-learning updates
            for i in range(len(states) - 1):
                state = states[i]
                action = actions[i]
                reward = rewards[i]
                next_state = states[i + 1]
                
                q = get_q(q_table, state)
                q[action] += ALPHA * 0.5 * (reward + GAMMA * np.max(get_q(q_table, next_state)) - q[action])

replay_buffer_red = ReplayBuffer(max_size=8)
replay_buffer_blue = ReplayBuffer(max_size=8)

def get_q(q_table, state):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS), dtype=float)
    return q_table[state]

# choose action while avoiding visited positions if possible
def choose_action(q_table, state, visited, epsilon=0.1, greedy=False):
    # build list of possible actions that don't step into visited cells
    possible_actions = []
    for i, (dx, dy) in enumerate(ACTIONS):
        nx = max(0, min(GRID_SIZE - 1, state[0] + dx))
        ny = max(0, min(GRID_SIZE - 1, state[1] + dy))
        if (nx, ny) not in visited:
            possible_actions.append(i)

    # if all neighbors visited, allow any action (prevents deadlock)
    if not possible_actions:
        possible_actions = list(range(len(ACTIONS)))

    q_vals = get_q(q_table, state)

    if greedy:
        # pick best among allowed
        return max(possible_actions, key=lambda a: q_vals[a])

    # epsilon-greedy among allowed actions
    if random.random() < epsilon:
        return random.choice(possible_actions)
    else:
        return max(possible_actions, key=lambda a: q_vals[a])

def step(position, action):
    x, y = position
    dx, dy = ACTIONS[action]
    new_x = max(0, min(GRID_SIZE - 1, x + dx))
    new_y = max(0, min(GRID_SIZE - 1, y + dy))
    return (new_x, new_y)

def calculate_reward(new_pos, visited, goal_pos):
    """Calculate reward based on position and visit history"""
    reward = -1.0  # Step penalty (increased from -0.01)
    
    if new_pos == goal_pos:
        reward = 100.0  # Large goal reward
    elif new_pos in visited:
        reward -= 5.0  # Strong penalty for revisiting
    
    return reward

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# -------- Pygame UI --------
pygame.init()
screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + 40))
pygame.display.set_caption("Two-Agent Q-learning (Red & Blue)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

def draw(red_pos, blue_pos, red_path, blue_path, ep_text):
    screen.fill((255, 255, 255))
    # grid
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (210, 210, 210), rect, 1)

    # draw visited path cells (lighter)
    for (rx, ry) in red_path:
        pygame.draw.rect(screen, (255, 200, 200), (ry * CELL_SIZE, rx * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    for (bx, by) in blue_path:
        pygame.draw.rect(screen, (200, 200, 255), (by * CELL_SIZE, bx * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # goal
    gx, gy = GOAL
    pygame.draw.rect(screen, (0, 200, 0), (gy * CELL_SIZE, gx * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # agents
    rx, ry = red_pos
    bx, by = blue_pos
    pygame.draw.rect(screen, (200, 0, 0), (ry * CELL_SIZE + 6, rx * CELL_SIZE + 6, CELL_SIZE - 12, CELL_SIZE - 12))
    pygame.draw.rect(screen, (0, 0, 200), (by * CELL_SIZE + 6, bx * CELL_SIZE + 6, CELL_SIZE - 12, CELL_SIZE - 12))

    # text
    txt = font.render(ep_text, True, (0, 0, 0))
    screen.blit(txt, (8, GRID_SIZE * CELL_SIZE + 8))
    pygame.display.flip()

# -------- Training loop --------
epsilon = EPSILON_START
history = []  # store (episode, steps_red, steps_blue)
best_red = None
best_blue = None

for ep in range(EPISODES):
    red_pos = RED_START
    blue_pos = BLUE_START
    visited_red = set([red_pos])
    visited_blue = set([blue_pos])
    red_path = [red_pos]
    blue_path = [blue_pos]
    steps_red = 0
    steps_blue = 0
    done_red = done_blue = False
    step_count = 0
    
    # Track episode data for replay buffer
    red_episode_data = {'states': [red_pos], 'actions': [], 'rewards': []}
    blue_episode_data = {'states': [blue_pos], 'actions': [], 'rewards': []}
    
    # Track progress for early termination
    red_last_distance = manhattan_distance(red_pos, GOAL)
    blue_last_distance = manhattan_distance(blue_pos, GOAL)
    red_no_progress_count = 0
    blue_no_progress_count = 0

    while not (done_red and done_blue) and step_count < MAX_STEPS:
        # event handling (so window remains responsive)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # RED
        if not done_red:
            s_r = red_pos
            a_r = choose_action(q_red, s_r, visited_red, epsilon=epsilon, greedy=False)
            new_r = step(s_r, a_r)
            reward_r = calculate_reward(new_r, visited_red, GOAL)
            
            # Store episode data
            red_episode_data['actions'].append(a_r)
            red_episode_data['rewards'].append(reward_r)
            red_episode_data['states'].append(new_r)
            
            # Check progress for early termination
            current_distance = manhattan_distance(new_r, GOAL)
            if current_distance >= red_last_distance:
                red_no_progress_count += 1
            else:
                red_no_progress_count = 0
            red_last_distance = current_distance
            
            # Force early termination if stuck in loops
            if red_no_progress_count > 10 and steps_red > MAX_STEPS // 2:
                done_red = True
                reward_r -= 10  # Additional penalty for getting stuck
            
            # Q update
            q = get_q(q_red, s_r)
            q[a_r] += ALPHA * (reward_r + GAMMA * np.max(get_q(q_red, new_r)) - q[a_r])
            red_pos = new_r
            visited_red.add(red_pos)
            red_path.append(red_pos)
            steps_red += 1
            if red_pos == GOAL:
                done_red = True

        # BLUE
        if not done_blue:
            s_b = blue_pos
            a_b = choose_action(q_blue, s_b, visited_blue, epsilon=epsilon, greedy=False)
            new_b = step(s_b, a_b)
            reward_b = calculate_reward(new_b, visited_blue, GOAL)
            
            # Store episode data
            blue_episode_data['actions'].append(a_b)
            blue_episode_data['rewards'].append(reward_b)
            blue_episode_data['states'].append(new_b)
            
            # Check progress for early termination
            current_distance = manhattan_distance(new_b, GOAL)
            if current_distance >= blue_last_distance:
                blue_no_progress_count += 1
            else:
                blue_no_progress_count = 0
            blue_last_distance = current_distance
            
            # Force early termination if stuck in loops
            if blue_no_progress_count > 10 and steps_blue > MAX_STEPS // 2:
                done_blue = True
                reward_b -= 10  # Additional penalty for getting stuck
            
            qb = get_q(q_blue, s_b)
            qb[a_b] += ALPHA * (reward_b + GAMMA * np.max(get_q(q_blue, new_b)) - qb[a_b])
            blue_pos = new_b
            visited_blue.add(blue_pos)
            blue_path.append(blue_pos)
            steps_blue += 1
            if blue_pos == GOAL:
                done_blue = True

        step_count += 1

        # draw (optionally) - if you want faster training set SHOW_TRAINING=False
        if SHOW_TRAINING:
            ep_text = f"Training Ep {ep+1}/{EPISODES}  Steps (R,B): {steps_red},{steps_blue}  Eps: {epsilon:.3f}"
            draw(red_pos, blue_pos, red_path, blue_path, ep_text)
            clock.tick(FPS)

    # end episode
    history.append((ep + 1, steps_red if steps_red>0 else MAX_STEPS, steps_blue if steps_blue>0 else MAX_STEPS))
    print(f"Episode {ep+1}: Red Steps={steps_red}, Blue Steps={steps_blue}, Epsilon={epsilon:.3f}")

    # Add successful episodes to replay buffer
    if red_pos == GOAL and steps_red <= 50:  # Only add reasonably efficient solutions
        replay_buffer_red.add_episode("red", red_episode_data)
    if blue_pos == GOAL and steps_blue <= 50:
        replay_buffer_blue.add_episode("blue", blue_episode_data)

    # Periodically train from replay buffer
    if ep > 0 and ep % 10 == 0:  # Every 10 episodes
        replay_buffer_red.sample_and_train(q_red, sample_size=3)
        replay_buffer_blue.sample_and_train(q_blue, sample_size=3)
        print(f"  -> Replay training: Red buffer size={len(replay_buffer_red.buffer)}, Blue buffer size={len(replay_buffer_blue.buffer)}")

    # keep best (lowest steps) for quick reference
    if steps_red > 0:
        if best_red is None or steps_red < best_red[1]:
            best_red = (ep + 1, steps_red, red_path.copy())
    if steps_blue > 0:
        if best_blue is None or steps_blue < best_blue[1]:
            best_blue = (ep + 1, steps_blue, blue_path.copy())

    # decay epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

# small pause then validate greedy policy
time.sleep(0.6)

def validate_greedy_policy(q_table, agent_name, start_pos, max_test_steps=100):
    """Test if the greedy policy can reach the goal"""
    pos = start_pos
    steps = 0
    visited = set()
    
    while pos != GOAL and steps < max_test_steps:
        if pos in visited:
            # Hit a loop, policy is not good enough
            return False, steps
        visited.add(pos)
        
        action = choose_action(q_table, pos, visited=set(), greedy=True)
        pos = step(pos, action)
        steps += 1
    
    success = (pos == GOAL)
    print(f"{agent_name} validation: {'SUCCESS' if success else 'FAILED'} in {steps} steps")
    return success, steps

# Validate both agents before replay
print("\nValidating learned policies...")
red_valid, red_test_steps = validate_greedy_policy(q_red, "Red", RED_START)
blue_valid, blue_test_steps = validate_greedy_policy(q_blue, "Blue", BLUE_START)

if not (red_valid and blue_valid):
    print("Warning: One or both agents failed validation. Consider more training episodes.")
else:
    print("Both agents successfully validated!")

# -------- REPLAY (greedy) --------
print("\nREPLAY: Running greedy policy using learned Q-tables...")
red_pos = RED_START
blue_pos = BLUE_START
red_path = [red_pos]
blue_path = [blue_pos]
steps_red = 0
steps_blue = 0
done_red = done_blue = False

while not (done_red and done_blue) and (steps_red < GRID_SIZE*GRID_SIZE and steps_blue < GRID_SIZE*GRID_SIZE):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    if not done_red:
        a_r = choose_action(q_red, red_pos, visited=set(), greedy=True)  # greedy ignores visited for replay
        red_pos = step(red_pos, a_r)
        red_path.append(red_pos)
        steps_red += 1
        if red_pos == GOAL:
            done_red = True

    if not done_blue:
        a_b = choose_action(q_blue, blue_pos, visited=set(), greedy=True)
        blue_pos = step(blue_pos, a_b)
        blue_path.append(blue_pos)
        steps_blue += 1
        if blue_pos == GOAL:
            done_blue = True

    ep_text = f"REPLAY (greedy)  Red Steps: {steps_red}  Blue Steps: {steps_blue}"
    draw(red_pos, blue_pos, red_path, blue_path, ep_text)
    clock.tick(FPS)

print(f"Replay finished — Red Steps={steps_red}, Blue Steps={steps_blue}")

# keep window open few seconds
time.sleep(2.0)
pygame.quit()
