"""Non-blocking camera mosaic preview for inference (pygame window).

Resizable window with a simple dark UI. Runs in a daemon thread so inference is not blocked.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from _dashboard_camera_session import DashboardCameraSession

log = logging.getLogger(__name__)

_WINDOW_TITLE = "XLeRobot — live cameras"
_LAYOUT = ("head", "left_wrist", "right_wrist")
_ROLE_TITLES = {
    "head": "Head",
    "left_wrist": "Left wrist",
    "right_wrist": "Right wrist",
}

# RGB palette
_COLOR_BG = (22, 24, 30)
_COLOR_HEADER = (16, 18, 24)
_COLOR_PANEL_BG = (32, 35, 44)
_COLOR_BORDER = (72, 78, 96)
_COLOR_BORDER_ACTIVE = (94, 168, 130)
_COLOR_TEXT = (220, 226, 236)
_COLOR_MUTED = (130, 138, 158)
_COLOR_ACCENT = (94, 200, 140)

_MIN_W, _MIN_H = 640, 480
_DEFAULT_W, _DEFAULT_H = 1280, 720


def display_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _rgb_to_surface(rgb: np.ndarray) -> Any:
    import pygame

    rgb = np.ascontiguousarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")


class CameraPreviewWindow:
    """Background mosaic of head / left_wrist / right_wrist streams."""

    def __init__(
        self,
        source: DashboardCameraSession | dict[str, Any],
        *,
        preview_fps: float = 15.0,
        tile_width: int = 426,  # unused; kept for API compat
        tile_height: int = 320,
        window_width: int = _DEFAULT_W,
        window_height: int = _DEFAULT_H,
    ) -> None:
        del tile_width, tile_height  # layout is computed from window size
        self._session = source if hasattr(source, "streams") else None
        self._lerobot_cams = source if isinstance(source, dict) else None
        self._period = 1.0 / max(preview_fps, 1.0)
        self._win_w = max(_MIN_W, int(window_width))
        self._win_h = max(_MIN_H, int(window_height))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not display_available():
            log.warning("No DISPLAY/WAYLAND_DISPLAY — camera preview disabled")
            return
        try:
            import pygame  # noqa: F401
        except ImportError:
            log.warning("pygame not installed — camera preview disabled (uv add pygame)")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="pi05-camera-preview",
            daemon=True,
        )
        self._thread.start()
        print(f"Camera preview: resizable pygame window ({self._win_w}×{self._win_h}).")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.5)
            self._thread = None

    def _fetch_rgb(self, role: str) -> np.ndarray | None:
        if self._session is not None:
            stream = self._session.streams[role]
            rgb = stream.last_rgb
            if rgb is None:
                rgb = stream.get_rgb(timeout=0.05)
            return None if rgb is None else np.asarray(rgb)
        assert self._lerobot_cams is not None
        try:
            frame = self._lerobot_cams[role].read_latest(max_age_ms=1000)
            return np.asarray(frame)
        except Exception:
            return None

    def _layout_panels(self, win_w: int, win_h: int) -> dict[str, Any]:
        import pygame

        header_h = 44
        footer_h = 28
        margin = 12
        gap = 10
        content = pygame.Rect(
            margin,
            header_h,
            max(1, win_w - 2 * margin),
            max(1, win_h - header_h - footer_h - margin),
        )
        head_h = max(80, int(content.height * 0.52))
        wrist_h = max(60, content.height - head_h - gap)
        head_rect = pygame.Rect(content.x, content.y, content.width, head_h)
        wrist_y = content.y + head_h + gap
        half_w = max(1, (content.width - gap) // 2)
        return {
            "header": pygame.Rect(0, 0, win_w, header_h),
            "footer": pygame.Rect(0, win_h - footer_h, win_w, footer_h),
            "head": head_rect,
            "left_wrist": pygame.Rect(content.x, wrist_y, half_w, wrist_h),
            "right_wrist": pygame.Rect(content.x + half_w + gap, wrist_y, half_w, wrist_h),
        }

    def _draw_panel(
        self,
        screen: Any,
        rect: Any,
        rgb: np.ndarray | None,
        title: str,
        *,
        title_font: Any,
        small_font: Any,
        has_signal: bool,
    ) -> None:
        import pygame

        border_color = _COLOR_BORDER_ACTIVE if has_signal else _COLOR_BORDER
        pygame.draw.rect(screen, border_color, rect, width=2, border_radius=8)
        inner = rect.inflate(-4, -4)
        pygame.draw.rect(screen, _COLOR_PANEL_BG, inner, border_radius=6)

        if rgb is not None and rgb.size > 0:
            try:
                frame = _rgb_to_surface(rgb)
                scaled = pygame.transform.smoothscale(frame, inner.size)
                # Clip to rounded panel (optional: blit to subsurface)
                screen.blit(scaled, inner.topleft)
            except Exception:
                rgb = None

        if rgb is None:
            msg = small_font.render("No signal", True, _COLOR_MUTED)
            screen.blit(msg, msg.get_rect(center=inner.center))

        # Title chip (top-left of panel)
        chip_pad = (10, 5)
        label = title_font.render(title, True, _COLOR_TEXT)
        chip = pygame.Surface(
            (label.get_width() + 2 * chip_pad[0], label.get_height() + 2 * chip_pad[1]),
            pygame.SRCALPHA,
        )
        chip.fill((*_COLOR_HEADER, 210))
        chip.blit(label, chip_pad)
        screen.blit(chip, (inner.x + 8, inner.y + 8))

    def _draw_chrome(
        self,
        screen: Any,
        rects: dict[str, Any],
        *,
        title_font: Any,
        small_font: Any,
    ) -> None:
        import pygame

        screen.fill(_COLOR_BG)
        pygame.draw.rect(screen, _COLOR_HEADER, rects["header"])
        title = title_font.render(_WINDOW_TITLE, True, _COLOR_TEXT)
        screen.blit(title, (16, rects["header"].centery - title.get_height() // 2))
        hint = small_font.render("drag corner to resize", True, _COLOR_MUTED)
        screen.blit(
            hint,
            (rects["header"].right - hint.get_width() - 16, rects["header"].centery - hint.get_height() // 2),
        )
        pygame.draw.rect(screen, _COLOR_HEADER, rects["footer"])
        foot = small_font.render(
            "Q or Esc — close preview only (inference keeps running)",
            True,
            _COLOR_MUTED,
        )
        screen.blit(foot, (16, rects["footer"].centery - foot.get_height() // 2))

    def _render_frame(self, screen: Any, fonts: tuple[Any, Any]) -> None:
        title_font, small_font = fonts
        rects = self._layout_panels(self._win_w, self._win_h)
        self._draw_chrome(screen, rects, title_font=title_font, small_font=small_font)
        for role in _LAYOUT:
            rgb = self._fetch_rgb(role)
            self._draw_panel(
                screen,
                rects[role],
                rgb,
                _ROLE_TITLES[role],
                title_font=small_font,
                small_font=small_font,
                has_signal=rgb is not None,
            )

    def _run(self) -> None:
        import pygame

        pygame.init()
        try:
            pygame.display.set_caption(_WINDOW_TITLE)
            screen = pygame.display.set_mode(
                (self._win_w, self._win_h),
                pygame.RESIZABLE,
            )
            title_font = pygame.font.SysFont("dejavusans,sans-serif", 20, bold=True)
            small_font = pygame.font.SysFont("dejavusans,sans-serif", 14)
            clock = pygame.time.Clock()
            user_closed = False

            while not self._stop.is_set() and not user_closed:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        user_closed = True
                        log.info("Camera preview closed (window quit; inference continues)")
                    elif event.type == pygame.KEYDOWN and event.key in (
                        pygame.K_q,
                        pygame.K_ESCAPE,
                    ):
                        user_closed = True
                        log.info("Camera preview closed by user (inference continues)")
                    elif event.type == pygame.VIDEORESIZE:
                        self._win_w = max(_MIN_W, event.w)
                        self._win_h = max(_MIN_H, event.h)
                        screen = pygame.display.set_mode(
                            (self._win_w, self._win_h),
                            pygame.RESIZABLE,
                        )

                try:
                    self._render_frame(screen, (title_font, small_font))
                    pygame.display.flip()
                except Exception as e:
                    log.debug("preview frame error: %s", e)

                clock.tick(max(1, int(round(1.0 / self._period))))
        finally:
            pygame.quit()
