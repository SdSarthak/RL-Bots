import json

import pytest

from rlbot.config import ACTIONS, Config, ConfigError


def test_defaults_match_documented_setup():
    cfg = Config()
    assert cfg.grid_size == 6
    assert cfg.goal == (5, 5)
    assert cfg.red_start == cfg.blue_start == (0, 0)
    assert len(ACTIONS) == 4


def test_positions_are_coerced_to_tuples():
    cfg = Config(goal=[2, 3], obstacles=[[1, 1], [1, 2]])
    assert cfg.goal == (2, 3)
    assert cfg.obstacles == ((1, 1), (1, 2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid_size": 1},
        {"goal": (9, 9)},
        {"alpha": 0.0},
        {"gamma": 1.5},
        {"epsilon_min": 0.9, "epsilon_start": 0.1},
        {"epsilon_decay": 0.0},
        {"episodes": 0},
        {"max_steps": 0},
        {"replay_frequency": 0},
        {"goal": (0, 0)},
        {"obstacles": [(5, 5)]},
        {"obstacles": [(0, 0)]},
    ],
)
def test_invalid_configs_are_rejected(kwargs):
    with pytest.raises(ConfigError):
        Config(**kwargs)


def test_unknown_keys_are_rejected():
    with pytest.raises(ConfigError) as excinfo:
        Config.from_dict({"grid_size": 5, "learning_rate": 0.1})
    assert "learning_rate" in str(excinfo.value)


def test_round_trips_through_json(tmp_path):
    cfg = Config(grid_size=7, goal=(6, 6), obstacles=[(1, 1)], seed=42)
    path = cfg.save(tmp_path / "cfg.json")
    loaded = Config.from_file(path)
    assert loaded == cfg
    assert json.loads(path.read_text(encoding="utf-8"))["obstacles"] == [[1, 1]]


def test_replace_returns_a_new_config():
    cfg = Config()
    other = cfg.replace(episodes=5)
    assert other.episodes == 5
    assert cfg.episodes == 80
    with pytest.raises(ConfigError):
        cfg.replace(nope=1)


def test_from_file_reports_bad_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        Config.from_file(path)


def test_from_file_reports_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        Config.from_file(tmp_path / "absent.json")


def test_shipped_configs_are_valid():
    from pathlib import Path

    configs = Path(__file__).resolve().parents[1] / "configs"
    files = sorted(configs.glob("*.json"))
    assert files, "expected example configs to ship with the repo"
    for path in files:
        Config.from_file(path)
