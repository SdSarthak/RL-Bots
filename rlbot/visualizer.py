"""Pygame rendering.

Pygame is imported lazily inside :meth:`PygameRenderer.open` so the rest of the
package stays importable on machines with no display, no SDL and no pygame
installed at all.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

from rlbot.config import COLORS, Config, Position
from rlbot.environment import GridWorld
from rlbot.trainer import Frame

AGENT_COLORS = {"red": COLORS["RED_AGENT"], "blue": COLORS["BLUE_AGENT"]}
TRAIL_COLORS = {"red": COLORS["RED_PATH"], "blue": COLORS["BLUE_PATH"]}
CAPTION_BAR_HEIGHT = 40


class DisplayUnavailable(RuntimeError):
    """Raised when pygame cannot be imported or no display can be opened."""


class PygameRenderer:
    """Draws the grid, trails and agents; usable as a context manager.

    Call the instance with a :class:`~rlbot.trainer.Frame` to draw it. It
    returns ``False`` once the user closes the window, which is the signal
    :func:`~rlbot.trainer.train` uses to stop early.
    """

    def __init__(self, env: GridWorld, config: Optional[Config] = None) -> None:
        self.env = env
        self.config = config or env.config
        self.pygame = None
        self.screen = None
        self.clock = None
        self.font = None
        self.closed = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def open(self) -> "PygameRenderer":
        try:
            import pygame  # noqa: PLC0415 - deliberately deferred
        except ImportError as exc:
            raise DisplayUnavailable(
                "pygame is not installed; run with --headless or `pip install pygame`"
            ) from exc

        try:
            pygame.init()
            size = self.config.grid_size * self.config.cell_size
            screen = pygame.display.set_mode((size, size + CAPTION_BAR_HEIGHT))
        except Exception as exc:  # pygame.error and SDL failures alike
            try:
                pygame.quit()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            raise DisplayUnavailable(
                f"could not open a display ({exc}); run with --headless"
            ) from exc

        pygame.display.set_caption("RL Bot - Two-Agent Q-learning")
        self.pygame = pygame
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.closed = False
        return self

    def close(self) -> None:
        if self.pygame is not None and not self.closed:
            self.pygame.quit()
        self.closed = True
        self.screen = None

    def __enter__(self) -> "PygameRenderer":
        return self.open()

    def __exit__(self, *_exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def _pump_events(self) -> bool:
        """Keep the window responsive; ``False`` means the user asked to quit."""
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
        return True

    def _cell_rect(self, pos: Position, inset: int = 0):
        cell = self.config.cell_size
        row, col = pos
        return (
            col * cell + inset,
            row * cell + inset,
            cell - 2 * inset,
            cell - 2 * inset,
        )

    def draw(self, frame: Frame) -> bool:
        """Render one frame. Returns ``False`` when the window was closed."""
        if self.screen is None:
            raise RuntimeError("renderer is not open; call open() first")
        pygame = self.pygame
        if not self._pump_events():
            return False

        self.screen.fill(COLORS["WHITE"])

        for name, path in frame.paths.items():
            colour = TRAIL_COLORS.get(name, COLORS["GRID"])
            for cell in path:
                pygame.draw.rect(self.screen, colour, self._cell_rect(cell))

        for obstacle in self.env.obstacles:
            pygame.draw.rect(self.screen, COLORS["OBSTACLE"], self._cell_rect(obstacle))

        pygame.draw.rect(self.screen, COLORS["GOAL"], self._cell_rect(self.env.goal))

        for row in range(self.config.grid_size):
            for col in range(self.config.grid_size):
                pygame.draw.rect(self.screen, COLORS["GRID"], self._cell_rect((row, col)), 1)

        for name, position in frame.positions.items():
            colour = AGENT_COLORS.get(name, COLORS["BLACK"])
            pygame.draw.rect(self.screen, colour, self._cell_rect(position, inset=6))

        text = self.font.render(frame.caption, True, COLORS["BLACK"])
        self.screen.blit(text, (8, self.config.grid_size * self.config.cell_size + 10))
        pygame.display.flip()
        self.clock.tick(self.config.fps)
        return True

    __call__ = draw


class AsciiRenderer:
    """Terminal fallback renderer, used by ``--render ascii``."""

    def __init__(self, env: GridWorld, every: int = 1) -> None:
        self.env = env
        self.every = max(1, every)
        self._count = 0

    def draw(self, frame: Frame) -> bool:
        self._count += 1
        if self._count % self.every:
            return True
        print(frame.caption)
        print(self.env.render_ascii(frame.positions))
        print()
        return True

    __call__ = draw

    def close(self) -> None:  # parity with PygameRenderer
        return None


def display_available() -> bool:
    """Best-effort check for a usable video device before opening a window."""
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        return False
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def replay_paths(
    renderer: "PygameRenderer | AsciiRenderer",
    paths: Dict[str, List[Position]],
    caption: str = "Replay (greedy policy)",
) -> bool:
    """Animate pre-computed paths, e.g. the greedy rollouts after training."""
    if not paths:
        return True
    longest = max(len(p) for p in paths.values())
    for i in range(longest):
        positions = {name: path[min(i, len(path) - 1)] for name, path in paths.items()}
        trails = {name: path[: min(i, len(path) - 1) + 1] for name, path in paths.items()}
        steps = "  ".join(f"{n}:{min(i, len(p) - 1)}" for n, p in paths.items())
        frame = Frame(positions=positions, paths=trails, caption=f"{caption}  {steps}")
        if renderer(frame) is False:
            return False
    return True
