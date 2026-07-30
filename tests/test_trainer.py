import csv
import json

import pytest

from rlbot.config import Config
from rlbot.environment import GridWorld
from rlbot.trainer import (
    Frame,
    format_summary,
    moving_average,
    rollout_greedy,
    starts_for,
    train,
    validate_policies,
)


def test_training_reaches_the_goal_optimally(config):
    result = train(config)
    assert len(result.history) == config.episodes
    for name in ("red", "blue"):
        optimal = result.env.optimal_steps(starts_for(config)[name])
        assert result.best[name].steps == optimal
        assert result.success_rate(name) > 0.5


def test_greedy_policy_solves_the_default_board():
    result = train(Config(seed=0))
    outcomes = validate_policies(result, verbose=False)
    for name, rollout in outcomes.items():
        assert rollout.reached_goal, f"{name} failed to reach the goal greedily"
        assert rollout.steps == result.env.optimal_steps(starts_for(result.config)[name])


def test_greedy_policy_solves_a_maze_with_obstacles():
    config = Config(
        grid_size=6,
        goal=(5, 5),
        obstacles=[(1, 1), (1, 2), (1, 3), (1, 4), (3, 1), (3, 2), (3, 3), (3, 4)],
        episodes=250,
        max_steps=200,
        alpha=0.3,
        gamma=0.97,
        epsilon_decay=0.98,
        epsilon_min=0.05,
        seed=11,
    )
    result = train(config)
    outcomes = validate_policies(result, verbose=False)
    assert all(r.reached_goal for r in outcomes.values())


def test_training_is_reproducible_for_a_fixed_seed(config):
    a = train(config)
    b = train(config)
    assert a.steps_series("red") == b.steps_series("red")
    assert a.steps_series("blue") == b.steps_series("blue")


def test_different_seeds_explore_differently(config):
    a = train(config)
    b = train(config.replace(seed=config.seed + 1))
    assert a.steps_series("red") != b.steps_series("red")


def test_agents_learn_to_take_fewer_steps(config):
    result = train(config.replace(episodes=60))
    steps = result.steps_series("red")
    first_third = sum(steps[:20]) / 20
    last_third = sum(steps[-20:]) / 20
    assert last_third < first_third


def test_episodes_respect_the_step_cap():
    config = Config(grid_size=6, goal=(5, 5), episodes=5, max_steps=8, seed=3)
    result = train(config)
    for stats in result.history:
        for rollout in stats.rollouts.values():
            assert rollout.steps <= config.max_steps


def test_unreachable_goal_is_rejected_before_training():
    config = Config(grid_size=3, goal=(2, 2), obstacles=[(1, 2), (2, 1), (1, 1)])
    with pytest.raises(ValueError, match="unreachable"):
        train(config)


def test_replay_buffer_runs_on_schedule(config):
    result = train(config.replace(replay_frequency=5, episodes=20))
    replayed = [s.episode for s in result.history if s.replay_updates]
    assert replayed
    assert all(ep % 5 == 0 for ep in replayed)


def test_renderer_receives_frames_and_can_stop_training(config):
    frames = []

    def renderer(frame: Frame) -> bool:
        frames.append(frame)
        return len(frames) < 5

    result = train(config, renderer=renderer)
    assert result.interrupted
    assert len(frames) == 5
    assert set(frames[0].positions) == {"red", "blue"}
    assert frames[0].caption.startswith("Train 1/")


def test_on_episode_end_callback_sees_every_episode(config):
    seen = []
    train(config.replace(episodes=7), on_episode_end=lambda s: seen.append(s.episode))
    assert seen == list(range(1, 8))


def test_rollout_greedy_stops_instead_of_looping():
    # An untrained agent has a flat Q-table, so the walk must still terminate.
    config = Config(grid_size=5, goal=(4, 4), max_test_steps=25, seed=5)
    result = train(config.replace(episodes=1))
    rollout = rollout_greedy(result.agents["red"], result.env, (0, 0), max_steps=25)
    assert rollout.steps <= 25
    assert rollout.reason in {"goal", "stuck", "max_steps", "loop"}


def test_rollout_greedy_on_a_hand_built_policy():
    config = Config(grid_size=3, goal=(0, 2))
    env = GridWorld(config)
    result = train(config.replace(episodes=1))
    agent = result.agents["red"]
    agent.q.clear()
    agent.q_values((0, 0))[0] = 1.0  # Right
    agent.q_values((0, 1))[0] = 1.0  # Right
    rollout = rollout_greedy(agent, env, (0, 0))
    assert rollout.path == [(0, 0), (0, 1), (0, 2)]
    assert rollout.reached_goal
    assert rollout.reason == "goal"
    assert rollout.total_reward == pytest.approx(config.step_penalty + config.goal_reward)


def test_summary_and_format(config):
    result = train(config.replace(episodes=10))
    summary = result.summary()
    assert set(summary) == {"red", "blue"}
    assert summary["red"]["episodes"] == 10
    assert summary["red"]["states_learned"] > 0
    text = format_summary(result)
    assert "red" in text and "blue" in text


def test_artifacts_are_written(tmp_path, config):
    result = train(config.replace(episodes=6))
    csv_path = result.save_history_csv(tmp_path / "runs" / "history.csv")
    json_path = result.save_summary_json(tmp_path / "runs" / "summary.json")

    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {"red_steps", "blue_steps", "epsilon"} <= set(rows[0])

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["red"]["episodes"] == 6
    assert payload["config"]["grid_size"] == config.grid_size


def test_moving_average():
    assert moving_average([1, 2, 3], window=1) == [1, 2, 3]
    assert moving_average([1, 2, 3, 4], window=2) == [1.0, 1.5, 2.5, 3.5]
    with pytest.raises(ValueError):
        moving_average([1, 2], window=0)


def test_rollout_serialisation(config):
    result = train(config.replace(episodes=3))
    data = result.history[0].to_dict()
    assert data["episode"] == 1
    assert "red" in data["agents"]
    assert isinstance(data["agents"]["red"]["path"], list)
