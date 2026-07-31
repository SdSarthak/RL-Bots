"""Boundary and failure-mode tests for config loading and Q-table loading.

Everything here uses hand-built values, so nothing downloads, trains for long,
or depends on a display.
"""

import json
import random

import numpy as np
import pytest

from rlbot.agent import QLearningAgent
from rlbot.config import (
    _BOOL_FIELDS,
    _FLOAT_FIELDS,
    _INT_FIELDS,
    ACTIONS,
    Config,
    ConfigError,
)
from rlbot.environment import GridWorld

POSITION_FIELDS = {"red_start", "blue_start", "goal", "obstacles"}


def make_agent(config=None, seed=0):
    config = config or Config(grid_size=4, goal=(3, 3))
    return QLearningAgent("red", GridWorld(config), config, rng=random.Random(seed))


# ----------------------------------------------------------------------
# config coercion
# ----------------------------------------------------------------------
def test_every_config_field_is_type_checked():
    """A new field that escapes coerce() would silently accept any garbage."""
    covered = set(_INT_FIELDS) | set(_FLOAT_FIELDS) | set(_BOOL_FIELDS) | POSITION_FIELDS
    assert set(Config.field_names()) == covered
    groups = [set(_INT_FIELDS), set(_FLOAT_FIELDS), set(_BOOL_FIELDS), POSITION_FIELDS]
    for i, first in enumerate(groups):
        for second in groups[i + 1:]:
            assert not first & second, "a field must belong to exactly one group"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid_size": "6"},        # JSON string where a number was meant
        {"grid_size": 6.5},        # fractional grid, blew up inside range() later
        {"grid_size": True},       # booleans are ints in Python; here they are typos
        {"episodes": 2.5},
        {"max_steps": None},
        {"seed": 1.5},
        {"alpha": "0.2"},
        {"gamma": [0.9]},
        {"show_training": "yes"},
        {"fps": "fast"},
    ],
)
def test_wrong_types_raise_config_error_not_type_error(kwargs):
    with pytest.raises(ConfigError):
        Config(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"goal_reward": float("nan")},
        {"step_penalty": float("inf")},
        {"revisit_penalty": float("-inf")},
        {"loop_penalty": float("nan")},
        {"replay_alpha_scale": float("nan")},
    ],
)
def test_non_finite_numbers_are_rejected(kwargs):
    """A single NaN reward propagates into every Q-value it ever touches."""
    with pytest.raises(ConfigError, match="finite"):
        Config(**kwargs)


def test_json_style_whole_floats_are_accepted():
    cfg = Config(grid_size=6.0, episodes=10.0, goal=[5.0, 5.0])
    assert cfg.grid_size == 6 and isinstance(cfg.grid_size, int)
    assert cfg.episodes == 10
    assert cfg.goal == (5, 5)


def test_fractional_positions_are_not_silently_truncated():
    with pytest.raises(ConfigError):
        Config(goal=(1.5, 2))
    with pytest.raises(ConfigError):
        Config(obstacles=[(1, 2.5)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replay_alpha_scale": 0.0},
        {"replay_alpha_scale": 1.5},
        {"efficient_solution_threshold": 0},
        {"no_progress_threshold": -1},
        {"max_test_steps": 0},
        {"cell_size": 0},
        {"fps": 0},
    ],
)
def test_out_of_range_values_are_rejected(kwargs):
    with pytest.raises(ConfigError):
        Config(**kwargs)


def test_a_badly_typed_config_file_is_reported_not_crashed(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"grid_size": "6"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="grid_size"):
        Config.from_file(path)


def test_nan_in_a_config_file_is_reported(tmp_path):
    # Python's json decoder happily accepts the bare NaN token.
    path = tmp_path / "nan.json"
    path.write_text('{"goal_reward": NaN}', encoding="utf-8")
    with pytest.raises(ConfigError, match="finite"):
        Config.from_file(path)


def test_replace_revalidates_types():
    with pytest.raises(ConfigError):
        Config().replace(episodes="many")


# ----------------------------------------------------------------------
# Q-table loading
# ----------------------------------------------------------------------
def test_q_table_from_a_bigger_board_is_rejected():
    """Loading an 8x8 table onto a 6x6 board used to succeed in silence."""
    agent = make_agent(Config(grid_size=4, goal=(3, 3)))
    with pytest.raises(ValueError, match="outside the 4x4 grid"):
        agent.load_state_dict({"q": {"7,7": [0.0, 0.0, 0.0, 0.0]}})


def test_q_table_with_negative_states_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        make_agent().load_state_dict({"q": {"-1,0": [0.0, 0.0, 0.0, 0.0]}})


def test_q_table_with_non_finite_values_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        make_agent().load_state_dict({"q": {"0,0": [0.0, float("nan"), 0.0, 0.0]}})


def test_q_table_with_non_numeric_values_is_rejected():
    with pytest.raises(ValueError, match="non-numeric"):
        make_agent().load_state_dict({"q": {"0,0": [{"a": 1}, 0.0, 0.0, 0.0]}})


def test_a_rejected_load_leaves_the_previous_table_intact():
    agent = make_agent()
    agent.q_values((0, 0))[0] = 7.0
    with pytest.raises(ValueError):
        agent.load_state_dict({"q": {"0,0": [1.0, 1.0, 1.0, 1.0], "9,9": [0, 0, 0, 0]}})
    assert set(agent.q) == {(0, 0)}
    assert agent.q[(0, 0)][0] == 7.0


def test_a_non_object_q_table_file_is_rejected(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        make_agent().load(path)


def test_q_table_round_trip_is_accepted_by_the_validator(tmp_path):
    agent = make_agent()
    agent.q_values((1, 2))[:] = [0.5, -0.5, 1.25, 0.0]
    path = agent.save(tmp_path / "red.json")
    restored = make_agent().load(path)
    assert np.allclose(restored.q[(1, 2)], agent.q[(1, 2)])


# ----------------------------------------------------------------------
# action selection under degenerate Q-values
# ----------------------------------------------------------------------
def test_best_action_reports_non_finite_values_instead_of_index_error():
    agent = make_agent()
    agent.q_values((0, 0))[:] = [float("nan"), 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="not finite"):
        agent.best_action((0, 0))


def test_best_action_rejects_an_out_of_range_action():
    with pytest.raises(IndexError):
        make_agent().best_action((0, 0), allowed=[len(ACTIONS)])


def test_ties_within_float_error_still_reach_the_rng():
    """Two routes of equal value must not both collapse onto action 0."""
    agent = make_agent(seed=1)
    row = agent.q_values((1, 1))
    row[:] = [1.0, 0.0, 1.0 + 5e-16, 0.0]  # Right and Down are the same number
    picked = {agent.best_action((1, 1), allowed=[0, 2]) for _ in range(60)}
    assert picked == {0, 2}


def test_distinct_values_are_never_treated_as_tied():
    agent = make_agent(seed=1)
    row = agent.q_values((1, 1))
    row[:] = [1.0, 0.0, 1.000001, 0.0]
    assert {agent.best_action((1, 1), allowed=[0, 2]) for _ in range(30)} == {2}
