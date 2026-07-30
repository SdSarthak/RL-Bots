import random

import pytest

from rlbot.agent import QLearningAgent
from rlbot.config import Config
from rlbot.environment import GridWorld
from rlbot.replay import Episode, ReplayBuffer

RIGHT, LEFT, DOWN, UP = 0, 1, 2, 3


def make_episode(name="red", steps=3, reward=1.0):
    episode = Episode(name)
    for i in range(steps):
        episode.add((0, i), RIGHT, reward, (0, i + 1), done=(i == steps - 1))
    episode.reached_goal = True
    return episode


def test_episode_tracks_length_and_reward():
    episode = make_episode(steps=4, reward=2.0)
    assert episode.steps == 4
    assert len(episode) == 4
    assert episode.total_reward == pytest.approx(8.0)
    assert episode.states() == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]


def test_empty_episode_has_no_states():
    assert Episode("red").states() == []


def test_buffer_rejects_empty_episodes():
    buffer = ReplayBuffer(max_size=4)
    assert buffer.add_episode(Episode("red")) is False
    assert len(buffer) == 0


def test_buffer_keeps_the_shortest_episodes():
    buffer = ReplayBuffer(max_size=2)
    buffer.add_episode(make_episode(steps=10))
    buffer.add_episode(make_episode(steps=3))
    buffer.add_episode(make_episode(steps=5))
    assert len(buffer) == 2
    assert buffer.episode_lengths() == [3, 5]
    assert buffer.best_episode().steps == 3


def test_buffer_reports_when_a_new_episode_was_evicted():
    buffer = ReplayBuffer(max_size=1)
    assert buffer.add_episode(make_episode(steps=2)) is True
    assert buffer.add_episode(make_episode(steps=9)) is False
    assert buffer.episode_lengths() == [2]


def test_invalid_buffer_size_raises():
    with pytest.raises(ValueError):
        ReplayBuffer(max_size=0)


def test_replay_is_a_no_op_below_the_minimum():
    buffer = ReplayBuffer(max_size=4, rng=random.Random(0))
    buffer.add_episode(make_episode(steps=2))
    agent = QLearningAgent("red", GridWorld(Config(grid_size=4, goal=(3, 3))))
    assert buffer.replay(agent) == 0
    assert agent.q == {}


def test_replay_propagates_value_backwards_in_one_pass():
    config = Config(grid_size=4, goal=(0, 3), alpha=1.0, gamma=1.0)
    env = GridWorld(config)
    agent = QLearningAgent("red", env, config, rng=random.Random(0))

    buffer = ReplayBuffer(max_size=4, rng=random.Random(0))
    for _ in range(2):
        episode = Episode("red")
        episode.add((0, 0), RIGHT, -1.0, (0, 1), done=False)
        episode.add((0, 1), RIGHT, -1.0, (0, 2), done=False)
        episode.add((0, 2), RIGHT, 100.0, (0, 3), done=True)
        buffer.add_episode(episode)

    updates = buffer.replay(agent, sample_size=1, alpha_scale=1.0)
    assert updates == 3
    # Reverse-order backups let the terminal reward reach the start immediately.
    assert agent.q_values((0, 2))[RIGHT] == pytest.approx(100.0)
    assert agent.q_values((0, 1))[RIGHT] == pytest.approx(99.0)
    assert agent.q_values((0, 0))[RIGHT] == pytest.approx(98.0)


def test_sample_never_exceeds_the_buffer():
    buffer = ReplayBuffer(max_size=5, rng=random.Random(1))
    for steps in (2, 3, 4):
        buffer.add_episode(make_episode(steps=steps))
    assert len(buffer.sample(10)) == 3
    assert buffer.sample(0) == []


def test_clear_empties_the_buffer():
    buffer = ReplayBuffer(max_size=3)
    buffer.add_episode(make_episode())
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.best_episode() is None
