"""HUD (Heads-Up Display) – renders telemetry and ADAS status panels.

The HUD is drawn entirely in screen space on top of the main renderer
output.  It uses a fixed-size left-side panel.
"""

from __future__ import annotations

import math


import pygame

from src import config
from src.sensors.radar import RadarReading


# Panel geometry.
_PANEL_X = 0
_PANEL_Y = 0
_PANEL_W = 270
_PANEL_H = config.WINDOW_HEIGHT
_PADDING = 16
_LINE_H = 22


class HUD:
    """Pygame HUD renderer.

    Args:
        screen: The target pygame surface to draw on.
    """

    def __init__(self, screen: pygame.Surface) -> None:
        self._screen = screen
        self._font_large = pygame.font.SysFont("monospace", 18, bold=True)
        self._font_small = pygame.font.SysFont("monospace", 14)
        self._font_title = pygame.font.SysFont("monospace", 20, bold=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        ego_speed_ms: float,
        target_speed_ms: float,
        acc_active: bool,
        lka_active: bool,
        lead_distance: float | None,
        radar_reading: RadarReading,
        steering_angle: float,
        acceleration: float,
        sim_time: float,
        warning: bool,
    ) -> None:
        """Render the full HUD panel.

        Args:
            ego_speed_ms:   Current ego speed (m/s).
            target_speed_ms: ACC target speed (m/s).
            acc_active:     True when ACC is engaged.
            lka_active:     True when LKA is engaged.
            lead_distance:  Distance to lead vehicle in metres, or None.
            radar_reading:  Latest radar scan result.
            steering_angle: Current steering angle (rad).
            acceleration:   Current acceleration (m/s²).
            sim_time:       Elapsed simulation time (s).
            warning:        True when a close-proximity warning is active.
        """
        # Semi-transparent background panel.
        panel = pygame.Surface((_PANEL_W, _PANEL_H), pygame.SRCALPHA)
        panel.fill((*config.COLOR_PANEL_BG, 210))
        self._screen.blit(panel, (_PANEL_X, _PANEL_Y))

        # Vertical border line.
        pygame.draw.line(
            self._screen,
            config.COLOR_PANEL_BORDER,
            (_PANEL_W, 0),
            (_PANEL_W, _PANEL_H),
            2,
        )

        y = _PADDING
        x = _PANEL_X + _PADDING

        # Title.
        self._draw_text("ADAS MONITOR", x, y, self._font_title, config.COLOR_EGO)
        y += _LINE_H + 6
        self._draw_separator(y)
        y += 10

        # Speed section.
        self._draw_label("SPEED", x, y)
        y += _LINE_H
        speed_kph = ego_speed_ms * 3.6
        target_kph = target_speed_ms * 3.6
        speed_color = (
            config.COLOR_WARNING if warning else config.COLOR_MARKING
        )
        self._draw_value(f"{speed_kph:5.1f} km/h", x, y, color=speed_color)
        y += _LINE_H
        self._draw_value(f"TGT {target_kph:5.1f} km/h", x, y)
        y += _LINE_H + 6

        # ADAS status section.
        self._draw_separator(y)
        y += 10
        self._draw_label("ADAS STATUS", x, y)
        y += _LINE_H
        self._draw_status("ACC", acc_active, x, y)
        y += _LINE_H
        self._draw_status("LKA", lka_active, x, y)
        y += _LINE_H + 6

        # Radar / proximity section.
        self._draw_separator(y)
        y += 10
        self._draw_label("RADAR", x, y)
        y += _LINE_H

        if lead_distance is not None:
            dist_color = (
                config.COLOR_WARNING
                if lead_distance < config.ACC_WARNING_DISTANCE_M
                else config.COLOR_UI_TEXT
            )
            self._draw_value(f"LEAD {lead_distance:5.1f} m", x, y, color=dist_color)
        else:
            self._draw_value("LEAD  -- m", x, y)
        y += _LINE_H + 6

        # Vehicle dynamics section.
        self._draw_separator(y)
        y += 10
        self._draw_label("DYNAMICS", x, y)
        y += _LINE_H
        steer_deg = math.degrees(steering_angle)
        self._draw_value(f"STEER {steer_deg:+6.1f} deg", x, y)
        y += _LINE_H
        self._draw_value(f"ACCEL {acceleration:+6.2f} m/s2", x, y)
        y += _LINE_H + 6

        # Simulation time.
        self._draw_separator(y)
        y += 10
        mins = int(sim_time) // 60
        secs = int(sim_time) % 60
        self._draw_value(f"T  {mins:02d}:{secs:02d}", x, y)
        y += _LINE_H + 12

        # Controls help.
        self._draw_separator(y)
        y += 10
        self._draw_label("CONTROLS", x, y)
        y += _LINE_H
        helps = [
            "UP/DN   target speed",
            "CLICK   place obstacle",
            "R       clear obstacles",
            "ESC     quit",
        ]
        for line in helps:
            self._draw_value(line, x, y, color=(90, 110, 125))
            y += _LINE_H

        # Warning banner.
        if warning:
            self._draw_warning_banner()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        surf = font.render(text, True, color)
        self._screen.blit(surf, (x, y))

    def _draw_label(self, text: str, x: int, y: int) -> None:
        self._draw_text(text, x, y, self._font_small, (100, 140, 170))

    def _draw_value(
        self,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int] = config.COLOR_UI_TEXT,
    ) -> None:
        self._draw_text(text, x, y, self._font_large, color)

    def _draw_status(self, label: str, active: bool, x: int, y: int) -> None:
        color = (50, 220, 100) if active else (120, 60, 60)
        status = "ON " if active else "OFF"
        self._draw_text(
            f"{label:<4} [{status}]", x, y, self._font_large, color
        )

    def _draw_separator(self, y: int) -> None:
        pygame.draw.line(
            self._screen,
            config.COLOR_PANEL_BORDER,
            (_PANEL_X + _PADDING, y),
            (_PANEL_X + _PANEL_W - _PADDING, y),
            1,
        )

    def _draw_warning_banner(self) -> None:
        """Flashing red banner at the bottom of the panel."""
        banner_h = 36
        banner_y = _PANEL_H - banner_h - 10
        banner = pygame.Surface((_PANEL_W - 2 * _PADDING, banner_h), pygame.SRCALPHA)
        banner.fill((*config.COLOR_WARNING, 200))
        self._screen.blit(banner, (_PADDING, banner_y))
        text = self._font_large.render("! COLLISION RISK !", True, (255, 255, 255))
        tw = text.get_width()
        self._screen.blit(text, ((_PANEL_W - tw) // 2, banner_y + 8))
