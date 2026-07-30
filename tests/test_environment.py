import pytest

from rlbot.config import ACTIONS, Config
from rlbot.environment import GridWorld

RIGHT, LEFT, DOWN, UP = 0, 1, 2, 3


def test_actions_match_expected_directions():
    assert ACTIONS[RIGHT] == (0, 1)
    assert ACTIONS[LEFT] == (0, -1)
    assert ACTIONS[DOWN] == (1, 0)
    assert ACTIONS[UP] == (-1, 0)


def test_step_moves_in_the_expected_direction(env):
    assert env.step((1, 1), RIGHT) == (1, 2)
    assert env.step((1, 1), LEFT) == (1, 0)
    assert env.step((1, 1), DOWN) == (2, 1)
    assert env.step((1, 1), UP) == (0, 1)


def test_walls_clamp_instead_of_wrapping(env):
    assert env.step((0, 0), UP) == (0, 0)
    assert env.step((0, 0), LEFT) == (0, 0)
    last = env.size - 1
    assert env.step((last, last), DOWN) == (last, last)
    assert env.step((last, last), RIGHT) == (last, last)


def test_invalid_action_raises(env):
    with pytest.raises(IndexError):
        env.step((0, 0), 4)


def test_obstacles_block_movement():
    env = GridWorld(Config(grid_size=4, goal=(3, 3), obstacles=[(1, 1)]))
    assert env.step((1, 0), RIGHT) == (1, 0)
    assert env.step((0, 1), DOWN) == (0, 1)
    assert env.step((1, 0), DOWN) == (2, 0)
    assert not env.is_walkable((1, 1))


def test_neighbours_exclude_walls_and_obstacles():
    env = GridWorld(Config(grid_size=4, goal=(3, 3), obstacles=[(0, 1)]))
    assert sorted(env.neighbours((0, 0))) == [(1, 0)]
    assert sorted(env.neighbours((1, 1))) == [(1, 0), (1, 2), (2, 1)]


def test_reward_structure(env):
    cfg = env.config
    assert env.reward(env.goal, visited=set()) == cfg.goal_reward
    assert env.reward((1, 1), visited=set()) == cfg.step_penalty
    assert env.reward((1, 1), visited={(1, 1)}) == cfg.step_penalty + cfg.revisit_penalty
    # Reaching the goal pays full price even if the goal was somehow seen before.
    assert env.reward(env.goal, visited={env.goal}) == cfg.goal_reward


def test_manhattan_distance():
    assert GridWorld.manhattan_distance((0, 0), (3, 4)) == 7
    assert GridWorld.manhattan_distance((2, 2), (2, 2)) == 0


def test_shortest_path_on_open_grid(env):
    path = env.shortest_path((0, 0))
    assert path[0] == (0, 0)
    assert path[-1] == env.goal
    assert env.optimal_steps((0, 0)) == 6  # 4x4 grid, corner to corner


def test_shortest_path_routes_around_obstacles():
    env = GridWorld(Config(grid_size=3, goal=(2, 2), obstacles=[(0, 1), (1, 1)]))
    assert env.optimal_steps((0, 0)) == 4
    path = env.shortest_path((0, 0))
    assert (0, 1) not in path and (1, 1) not in path


def test_walled_off_goal_is_reported_unreachable():
    env = GridWorld(Config(grid_size=3, goal=(2, 2), obstacles=[(1, 2), (2, 1), (1, 1)]))
    assert env.shortest_path((0, 0)) is None
    assert env.optimal_steps((0, 0)) is None
    assert not env.is_solvable([(0, 0)])


def test_terminal_only_at_goal(env):
    assert env.is_terminal(env.goal)
    assert not env.is_terminal((0, 0))


def test_render_ascii_marks_agents_obstacles_and_goal():
    env = GridWorld(Config(grid_size=3, goal=(2, 2), obstacles=[(1, 1)]))
    board = env.render_ascii({"red": (0, 0)})
    assert board.splitlines() == ["R . .", ". # .", ". . G"]


def test_walkable_cells_excludes_obstacles():
    env = GridWorld(Config(grid_size=3, goal=(2, 2), obstacles=[(1, 1)]))
    cells = env.walkable_cells()
    assert len(cells) == 8
    assert (1, 1) not in cells
