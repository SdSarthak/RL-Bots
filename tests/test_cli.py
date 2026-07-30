import json

import pytest

from rlbot.cli import build_parser, config_from_args, main, normalise_argv


def parse(argv):
    return build_parser().parse_args(normalise_argv(argv))


def test_bare_flags_default_to_the_train_command():
    assert normalise_argv([]) == ["train"]
    assert normalise_argv(["--headless"]) == ["train", "--headless"]
    assert normalise_argv(["show"]) == ["show"]
    assert normalise_argv(["replay", "--load-dir", "x"]) == ["replay", "--load-dir", "x"]


def test_cli_overrides_beat_the_config_file(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"grid_size": 5, "goal": [4, 4], "episodes": 3}), encoding="utf-8")
    args = parse(["train", "--config", str(path), "--episodes", "9", "--seed", "5"])
    config = config_from_args(args)
    assert config.grid_size == 5
    assert config.episodes == 9
    assert config.seed == 5


def test_shrinking_the_grid_moves_the_default_goal():
    config = config_from_args(parse(["train", "--grid-size", "4"]))
    assert config.grid_size == 4
    assert config.goal == (3, 3)
    assert config.red_start == (0, 0)


def test_explicit_goal_is_respected():
    config = config_from_args(parse(["train", "--grid-size", "5", "--goal", "0", "4"]))
    assert config.goal == (0, 4)


def test_growing_the_grid_leaves_the_goal_alone():
    config = config_from_args(parse(["train", "--grid-size", "10"]))
    assert config.goal == (5, 5)


def test_headless_disables_training_visualisation():
    config = config_from_args(parse(["train", "--headless"]))
    assert config.show_training is False


def test_show_command_prints_the_board(capsys):
    assert main(["show", "--headless", "--grid-size", "4"]) == 0
    out = capsys.readouterr().out
    assert "Grid 4x4" in out
    assert "steps optimal" in out


def test_train_command_succeeds_and_saves_artifacts(tmp_path, capsys):
    save_dir = tmp_path / "run"
    code = main([
        "train", "--headless", "--quiet", "--episodes", "60", "--seed", "0",
        "--save-dir", str(save_dir),
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "SUCCESS" in out
    for name in ("red.json", "blue.json", "history.csv", "summary.json", "config.json"):
        assert (save_dir / name).exists(), name


def test_replay_command_reuses_saved_tables(tmp_path, capsys):
    save_dir = tmp_path / "run"
    assert main([
        "train", "--headless", "--quiet", "--episodes", "60", "--seed", "0",
        "--save-dir", str(save_dir),
    ]) == 0
    capsys.readouterr()

    assert main(["replay", "--headless", "--load-dir", str(save_dir)]) == 0
    out = capsys.readouterr().out
    assert "reached goal" in out


def test_replay_reports_missing_tables(tmp_path, capsys):
    assert main(["replay", "--headless", "--load-dir", str(tmp_path)]) == 2
    assert "Cannot replay" in capsys.readouterr().err


def test_train_rejects_an_unreachable_goal(tmp_path, capsys):
    path = tmp_path / "walled.json"
    path.write_text(
        json.dumps(
            {
                "grid_size": 3,
                "goal": [2, 2],
                "obstacles": [[1, 1], [1, 2], [2, 1]],
                "episodes": 2,
            }
        ),
        encoding="utf-8",
    )
    assert main(["train", "--headless", "--config", str(path)]) == 2
    assert "unreachable" in capsys.readouterr().err


def test_bad_config_returns_an_error_code(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"alpha": 0}', encoding="utf-8")
    assert main(["train", "--headless", "--config", str(path)]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_ascii_renderer_writes_to_stdout(capsys):
    assert main(["train", "--render", "ascii", "--quiet", "--episodes", "2",
                 "--grid-size", "4", "--no-replay-demo"]) in (0, 1)
    out = capsys.readouterr().out
    assert "Train 1/2" in out


@pytest.mark.parametrize("argv", [["--help"], ["train", "--help"]])
def test_help_exits_cleanly(argv):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 0
