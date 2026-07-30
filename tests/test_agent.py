import random

import numpy as np
import pytest

from rlbot.agent import QLearningAgent
from rlbot.config import ACTIONS, Config
from rlbot.environment import GridWorld

RIGHT, LEFT, DOWN, UP = 0, 1, 2, 3


def make_agent(config=None, seed=0):
    config = config or Config(grid_size=4, goal=(3, 3))
    env = GridWorld(config)
    return QLearningAgent("red", env, config, rng=random.Random(seed))


def test_q_rows_are_created_lazily_and_zeroed():
    agent = make_agent()
    assert agent.q == {}
    row = agent.q_values((0, 0))
    assert row.shape == (len(ACTIONS),)
    assert np.all(row == 0.0)
    assert (0, 0) in agent.q
    assert agent.q_values((0, 0)) is row


def test_best_action_picks_the_highest_value():
    agent = make_agent()
    agent.q_values((1, 1))[DOWN] = 5.0
    assert agent.best_action((1, 1)) == DOWN


def test_best_action_respects_the_allowed_set():
    agent = make_agent()
    agent.q_values((1, 1))[DOWN] = 5.0
    agent.q_values((1, 1))[UP] = 1.0
    assert agent.best_action((1, 1), allowed=[LEFT, UP]) == UP


def test_allowed_actions_skip_walls_and_visited_cells():
    agent = make_agent()
    # From the corner, up and left do not move, so they are never offered.
    assert sorted(agent.allowed_actions((0, 0))) == sorted([RIGHT, DOWN])
    # With (0, 1) already visited, only the downward move stays preferred.
    assert agent.allowed_actions((0, 0), visited={(0, 0), (0, 1)}) == [DOWN]


def test_allowed_actions_fall_back_when_everything_is_visited():
    agent = make_agent()
    visited = {(0, 0), (0, 1), (1, 0)}
    allowed = agent.allowed_actions((0, 0), visited=visited)
    assert sorted(allowed) == sorted([RIGHT, DOWN])


def test_boxed_in_agent_still_returns_an_action():
    config = Config(grid_size=3, goal=(2, 2), obstacles=[(0, 1), (1, 0)])
    agent = make_agent(config)
    allowed = agent.allowed_actions((0, 0), visited={(0, 0)})
    assert allowed  # never empty, otherwise the training loop would deadlock


def test_terminal_update_does_not_bootstrap():
    config = Config(grid_size=4, goal=(3, 3), alpha=1.0, gamma=0.9)
    agent = make_agent(config)
    agent.q_values((3, 3))[:] = 100.0  # a poisoned successor value
    agent.update((3, 2), RIGHT, reward=10.0, next_state=(3, 3), done=True)
    assert agent.q_values((3, 2))[RIGHT] == pytest.approx(10.0)


def test_non_terminal_update_bootstraps_from_the_successor():
    config = Config(grid_size=4, goal=(3, 3), alpha=1.0, gamma=0.9)
    agent = make_agent(config)
    agent.q_values((2, 2))[:] = 10.0
    agent.update((1, 2), DOWN, reward=-1.0, next_state=(2, 2), done=False)
    assert agent.q_values((1, 2))[DOWN] == pytest.approx(-1.0 + 0.9 * 10.0)


def test_update_returns_the_td_error():
    config = Config(grid_size=4, goal=(3, 3), alpha=0.5, gamma=0.0)
    agent = make_agent(config)
    td = agent.update((0, 0), RIGHT, reward=4.0, next_state=(0, 1), done=False)
    assert td == pytest.approx(4.0)
    assert agent.q_values((0, 0))[RIGHT] == pytest.approx(2.0)


def test_alpha_scale_dampens_the_update():
    config = Config(grid_size=4, goal=(3, 3), alpha=1.0, gamma=0.0)
    agent = make_agent(config)
    agent.update((0, 0), RIGHT, reward=8.0, next_state=(0, 1), done=False, alpha_scale=0.25)
    assert agent.q_values((0, 0))[RIGHT] == pytest.approx(2.0)


def test_greedy_selection_ignores_epsilon():
    agent = make_agent(seed=3)
    agent.q_values((1, 1))[UP] = 9.0
    for _ in range(20):
        assert agent.select_action((1, 1), epsilon=1.0, greedy=True) == UP


def test_epsilon_zero_is_deterministic():
    agent = make_agent(seed=3)
    agent.q_values((1, 1))[LEFT] = 3.0
    assert agent.select_action((1, 1), epsilon=0.0) == LEFT


def test_same_seed_gives_the_same_action_sequence():
    a = make_agent(seed=99)
    b = make_agent(seed=99)
    seq_a = [a.select_action((1, 1), epsilon=1.0) for _ in range(50)]
    seq_b = [b.select_action((1, 1), epsilon=1.0) for _ in range(50)]
    assert seq_a == seq_b


def test_policy_names_the_greedy_action():
    agent = make_agent()
    agent.q_values((0, 0))[DOWN] = 1.0
    assert agent.policy()[(0, 0)] == "Down"


def test_q_table_round_trips_through_disk(tmp_path):
    agent = make_agent()
    agent.q_values((0, 0))[RIGHT] = 1.5
    agent.q_values((2, 1))[UP] = -0.25
    path = agent.save(tmp_path / "red.json")

    restored = make_agent()
    restored.load(path)
    assert set(restored.q) == set(agent.q)
    for state, values in agent.q.items():
        assert np.allclose(restored.q[state], values)


def test_loading_a_malformed_table_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"q": {"0,0": [1.0, 2.0]}}', encoding="utf-8")
    with pytest.raises(ValueError):
        make_agent().load(path)


def test_loading_a_bad_state_key_raises(tmp_path):
    path = tmp_path / "bad_key.json"
    path.write_text('{"q": {"origin": [0, 0, 0, 0]}}', encoding="utf-8")
    with pytest.raises(ValueError):
        make_agent().load(path)


def test_loading_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_agent().load(tmp_path / "nope.json")
