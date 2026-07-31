"""Seeding, reproducibility and policy-consistency checks.

`--seed` is only worth having if it pins *everything*, so these tests compare
whole Q-tables and the on-disk artifacts rather than a summary statistic.
"""

import random

import pytest

from rlbot.agent import QLearningAgent
from rlbot.config import ACTION_NAMES, Config
from rlbot.environment import GridWorld
from rlbot.trainer import rollout_greedy, starts_for, train

SMALL = Config(grid_size=4, goal=(3, 3), episodes=30, max_steps=50, seed=1234)

MAZE = Config(
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


def q_tables(result):
    return {
        name: {state: list(map(float, row)) for state, row in sorted(agent.q.items())}
        for name, agent in result.agents.items()
    }


# ----------------------------------------------------------------------
# reproducibility
# ----------------------------------------------------------------------
def test_the_same_seed_reproduces_every_q_value():
    """steps_series matching is weak evidence; the tables must be identical."""
    assert q_tables(train(SMALL)) == q_tables(train(SMALL))


def test_a_different_seed_produces_a_different_table():
    assert q_tables(train(SMALL)) != q_tables(train(SMALL.replace(seed=SMALL.seed + 1)))


def test_the_same_seed_reproduces_the_history_artifacts(tmp_path):
    first = train(SMALL).save_history_csv(tmp_path / "a.csv").read_bytes()
    second = train(SMALL).save_history_csv(tmp_path / "b.csv").read_bytes()
    assert first == second


def test_seeding_does_not_leak_through_the_global_rng():
    """train() must not depend on module-level random state set by a caller."""
    random.seed(1)
    baseline = q_tables(train(SMALL))
    random.seed(999)
    [random.random() for _ in range(50)]
    assert q_tables(train(SMALL)) == baseline


def test_a_saved_table_reproduces_the_greedy_rollout(tmp_path):
    result = train(SMALL)
    env, start = result.env, starts_for(SMALL)["red"]
    before = rollout_greedy(result.agents["red"], env, start)

    path = result.agents["red"].save(tmp_path / "red.json")
    restored = QLearningAgent("red", env, SMALL, rng=random.Random(SMALL.seed))
    restored.load(path)
    after = rollout_greedy(restored, env, start)

    assert after.path == before.path
    assert after.reached_goal == before.reached_goal


def test_agents_do_not_share_a_random_stream():
    result = train(SMALL)
    red, blue = result.agents["red"], result.agents["blue"]
    assert red.rng is not blue.rng
    assert [red.rng.random() for _ in range(10)] != [blue.rng.random() for _ in range(10)]


# ----------------------------------------------------------------------
# the reported policy must be the policy that runs
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "config",
    [
        MAZE,
        # Undertrained on purpose: while the goal reward has not reached a
        # state, every real move there is negative and the untouched wall
        # actions still sit at zero. A raw argmax named a wall move for 22 of
        # this run's 63 states.
        Config(grid_size=8, goal=(7, 7), episodes=8, max_steps=120, seed=3),
    ],
    ids=["converged", "undertrained"],
)
def test_reported_policy_only_names_moves_the_agent_can_make(config):
    result = train(config)
    env = result.env
    for agent in result.agents.values():
        policy = agent.policy()
        assert policy
        for state, name in policy.items():
            action = ACTION_NAMES.index(name)
            assert env.step(state, action) != state, (
                f"{agent.name} policy at {state} says {name}, which walks into a wall"
            )


def test_reported_policy_matches_greedy_action_selection():
    """policy() and select_action(greedy=True) must not disagree."""
    result = train(MAZE)
    for agent in result.agents.values():
        for state, name in agent.policy().items():
            values = agent.q_values(state)
            chosen = agent.select_action(state, greedy=True)
            assert float(values[ACTION_NAMES.index(name)]) == pytest.approx(
                float(values[chosen])
            )


def test_policy_prefers_a_real_move_over_an_untouched_wall_action():
    """Wall actions keep their zero init while real moves go negative."""
    config = Config(grid_size=3, goal=(2, 2))
    agent = QLearningAgent("red", GridWorld(config), config, rng=random.Random(0))
    row = agent.q_values((0, 0))
    row[:] = [-1.0, 0.0, -2.0, 0.0]  # Right/Down explored, Left/Up never legal
    assert agent.policy()[(0, 0)] == "Right"


# ----------------------------------------------------------------------
# reward accounting
# ----------------------------------------------------------------------
def test_a_wall_bump_is_not_charged_a_step_penalty():
    """A boxed-in agent never moves, so its reward must stay at zero."""
    config = Config(grid_size=3, goal=(2, 2), obstacles=[(0, 1), (1, 0)])
    env = GridWorld(config)
    agent = QLearningAgent("red", env, config, rng=random.Random(0))

    stuck = rollout_greedy(agent, env, (0, 0), max_steps=5)

    assert stuck.reason == "stuck"
    assert stuck.path == [(0, 0)]
    assert stuck.steps == 0
    assert stuck.reached_goal is False
    # Previously this charged step_penalty + revisit_penalty for a move that
    # never happened, so a stuck rollout looked like it had walked somewhere.
    assert stuck.total_reward == 0.0


def test_reaching_the_goal_still_collects_the_goal_reward():
    config = Config(grid_size=3, goal=(0, 2))
    env = GridWorld(config)
    agent = QLearningAgent("red", env, config, rng=random.Random(0))
    agent.q_values((0, 0))[0] = 1.0
    agent.q_values((0, 1))[0] = 1.0
    rollout = rollout_greedy(agent, env, (0, 0))
    assert rollout.total_reward == pytest.approx(config.step_penalty + config.goal_reward)


# ----------------------------------------------------------------------
# train() argument checking
# ----------------------------------------------------------------------
def test_train_rejects_an_env_from_a_different_board():
    """env supplies the dynamics, config the rewards; they must agree."""
    config = Config(grid_size=4, goal=(3, 3), episodes=1)
    other = GridWorld(Config(grid_size=4, goal=(3, 3), obstacles=[(1, 1)]))
    with pytest.raises(ValueError, match="different board"):
        train(config, env=other)


def test_train_accepts_a_matching_env_built_separately():
    config = Config(grid_size=4, goal=(3, 3), episodes=2)
    env = GridWorld(Config(grid_size=4, goal=(3, 3), episodes=2))
    assert len(train(config, env=env).history) == 2


def test_train_rejects_agents_it_has_no_start_position_for():
    config = Config(grid_size=3, goal=(2, 2), episodes=1)
    env = GridWorld(config)
    agents = {"green": QLearningAgent("green", env, config, rng=random.Random(0))}
    with pytest.raises(ValueError, match="no start position"):
        train(config, env=env, agents=agents)
