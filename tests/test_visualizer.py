import os

import pytest

from rlbot.config import Config
from rlbot.environment import GridWorld
from rlbot.trainer import Frame
from rlbot.visualizer import AsciiRenderer, display_available, replay_paths

pygame = pytest.importorskip("pygame", reason="pygame is optional")

from rlbot.visualizer import PygameRenderer  # noqa: E402


@pytest.fixture
def dummy_video(monkeypatch):
    """Force SDL's headless driver so drawing works without a real display."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")


@pytest.fixture
def env():
    return GridWorld(Config(grid_size=5, goal=(4, 4), obstacles=[(2, 2)]))


def make_frame(caption="test"):
    return Frame(
        positions={"red": (0, 0), "blue": (1, 1)},
        paths={"red": [(0, 0)], "blue": [(0, 0), (1, 0), (1, 1)]},
        caption=caption,
    )


def test_pygame_renderer_draws_a_frame(dummy_video, env):
    with PygameRenderer(env) as renderer:
        assert renderer.draw(make_frame()) is True


def test_pygame_renderer_stops_on_quit_event(dummy_video, env):
    with PygameRenderer(env) as renderer:
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        assert renderer.draw(make_frame()) is False


def test_pygame_renderer_stops_on_escape(dummy_video, env):
    with PygameRenderer(env) as renderer:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        assert renderer.draw(make_frame()) is False


def test_drawing_before_open_is_an_error(env):
    with pytest.raises(RuntimeError):
        PygameRenderer(env).draw(make_frame())


def test_replay_paths_walks_every_step(dummy_video, env):
    with PygameRenderer(env) as renderer:
        paths = {"red": [(0, 0), (1, 0), (2, 0)], "blue": [(4, 4)]}
        assert replay_paths(renderer, paths) is True


def test_ascii_renderer_prints_the_board(env, capsys):
    renderer = AsciiRenderer(env, every=1)
    assert renderer.draw(make_frame("caption here")) is True
    renderer.close()
    out = capsys.readouterr().out
    assert "caption here" in out
    assert "#" in out  # the obstacle
    assert "G" in out


def test_ascii_renderer_honours_the_every_option(env, capsys):
    renderer = AsciiRenderer(env, every=3)
    for _ in range(3):
        renderer.draw(make_frame())
    assert capsys.readouterr().out.count("R") == 1


def test_replay_paths_accepts_no_paths(env):
    assert replay_paths(AsciiRenderer(env), {}) is True


def test_display_unavailable_under_the_dummy_driver(dummy_video):
    assert display_available() is False


def test_display_probe_reads_the_environment(monkeypatch):
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    if os.name == "nt":
        assert display_available() is True
