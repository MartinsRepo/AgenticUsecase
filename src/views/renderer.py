"""Pygame-based 2-D top-down renderer for the highway simulation.

The renderer converts world coordinates into screen coordinates using a
camera that follows the ego vehicle vertically.  The ego vehicle is always
drawn at a fixed vertical position on the screen (EGO_SCREEN_Y), and the
road scrolls beneath it.

Coordinate mapping
------------------
    screen_x = ROAD_CENTER_X + (world_x - ego_x) * PIXELS_PER_METER
    screen_y = EGO_SCREEN_Y  - (world_y - ego_y) * PIXELS_PER_METER

Positive world_y corresponds to *upward* on screen (the vehicle moves
toward the top of the window as it accelerates).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import pygame
import numpy as np

from src import config
from src.models.obstacle import NPCVehicle, Obstacle
from src.models.vehicle import Vehicle
from src.sensors.radar import RadarReading


class Renderer:
    """Draws every frame of the simulation.

    Args:
        screen: The main pygame display surface.
    """

    def __init__(self, screen: pygame.Surface) -> None:
        self._screen = screen
        self._w = screen.get_width()
        self._h = screen.get_height()

        # Pre-build a surface for the radar cone (supports alpha).
        self._radar_surf: pygame.Surface = pygame.Surface(
            (self._w, self._h), pygame.SRCALPHA
        )

    # ------------------------------------------------------------------
    # Public draw entry point
    # ------------------------------------------------------------------

    def draw_frame(
        self,
        ego: Vehicle,
        npcs: List[NPCVehicle],
        obstacles: List[Obstacle],
        radar_reading: RadarReading,
        warning: bool,
    ) -> None:
        """Render a complete frame.

        Args:
            ego:           Ego vehicle state.
            npcs:          List of NPC vehicles.
            obstacles:     List of static obstacles.
            radar_reading: Latest radar scan result.
            warning:       True when a close-proximity warning is active.
        """
        # Camera reference point.
        cam_x = ego.x
        cam_y = ego.y

        # 1. Background (offroad / grass).
        self._screen.fill(config.COLOR_OFFROAD)

        # 2. Road surface.
        self._draw_road(cam_x, cam_y)

        # 3. Radar cone (drawn before vehicles so it appears underneath them).
        self._draw_radar_cone(ego, cam_x, cam_y, radar_reading)

        # 4. Road markings (dashes).
        self._draw_lane_markings(cam_x, cam_y)

        # 5. NPC vehicles.
        for npc in npcs:
            self._draw_vehicle_box(
                npc.x, npc.y, 0.0, npc.width, npc.length, npc.color, cam_x, cam_y
            )

        # 6. Static obstacles.
        for obs in obstacles:
            self._draw_vehicle_box(
                obs.x, obs.y, 0.0, obs.width, obs.length, obs.color, cam_x, cam_y
            )

        # 7. Ego vehicle.
        ego_color = config.COLOR_WARNING if warning else config.COLOR_EGO
        self._draw_vehicle_box(
            ego.x, ego.y, ego.theta, ego.width, ego.length, ego_color, cam_x, cam_y,
            is_ego=True,
        )

        # 8. Speed-o-meter arc in top-right corner.
        self._draw_speedometer(ego.v)

    # ------------------------------------------------------------------
    # Road
    # ------------------------------------------------------------------

    def _draw_road(self, cam_x: float, cam_y: float) -> None:
        """Fill the road rectangle with asphalt colour."""
        road_rect = pygame.Rect(
            config.ROAD_LEFT_X, 0, config.ROAD_WIDTH_PX, self._h
        )
        pygame.draw.rect(self._screen, config.COLOR_ASPHALT, road_rect)

        # Shoulder strips (slightly lighter than asphalt).
        shoulder_w = int(config.SHOULDER_WIDTH_M * config.PIXELS_PER_METER)
        left_shoulder = pygame.Rect(config.ROAD_LEFT_X, 0, shoulder_w, self._h)
        right_shoulder = pygame.Rect(
            config.ROAD_RIGHT_X - shoulder_w, 0, shoulder_w, self._h
        )
        pygame.draw.rect(self._screen, config.COLOR_SHOULDER, left_shoulder)
        pygame.draw.rect(self._screen, config.COLOR_SHOULDER, right_shoulder)

        # Solid white edge lines.
        inner_left = config.ROAD_LEFT_X + shoulder_w
        inner_right = config.ROAD_RIGHT_X - shoulder_w
        pygame.draw.line(
            self._screen, config.COLOR_MARKING, (inner_left, 0), (inner_left, self._h), 3
        )
        pygame.draw.line(
            self._screen, config.COLOR_MARKING, (inner_right, 0), (inner_right, self._h), 3
        )

    def _draw_lane_markings(self, cam_x: float, cam_y: float) -> None:
        """Draw dashed centre-line markings between lanes."""
        dash_len = int(config.DASH_LENGTH_M * config.PIXELS_PER_METER)
        gap_len = int(config.DASH_GAP_M * config.PIXELS_PER_METER)
        period = dash_len + gap_len

        shoulder_w = int(config.SHOULDER_WIDTH_M * config.PIXELS_PER_METER)

        for lane_idx in range(1, config.NUM_LANES):
            # World-x of the lane divider.
            wx = (
                -config.NUM_LANES * config.LANE_WIDTH_M / 2.0
                + lane_idx * config.LANE_WIDTH_M
            )
            sx = self._world_to_screen_x(wx, cam_x)
            if sx < 0 or sx > self._w:
                continue

            # Compute the phase of the dash pattern based on camera position.
            phase = (cam_y * config.PIXELS_PER_METER) % period

            # Starting screen_y offset so dashes are aligned to world position.
            y_start = -int(phase)

            while y_start < self._h:
                y_end = min(y_start + dash_len, self._h)
                if y_end > 0:
                    pygame.draw.line(
                        self._screen,
                        config.COLOR_MARKING,
                        (sx, max(0, y_start)),
                        (sx, y_end),
                        2,
                    )
                y_start += period

    # ------------------------------------------------------------------
    # Vehicles / obstacles
    # ------------------------------------------------------------------

    def _draw_vehicle_box(
        self,
        wx: float,
        wy: float,
        theta: float,
        width: float,
        length: float,
        color: Tuple[int, int, int],
        cam_x: float,
        cam_y: float,
        is_ego: bool = False,
    ) -> None:
        """Draw a rotated vehicle rectangle with windshield highlight."""
        half_w = width / 2.0
        half_l = length / 2.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # World corners → screen corners.
        local_corners = [
            (-half_w,  half_l),
            ( half_w,  half_l),
            ( half_w, -half_l),
            (-half_w, -half_l),
        ]
        screen_pts = []
        for lx, ly in local_corners:
            world_cx = wx + lx * cos_t + ly * sin_t
            world_cy = wy - lx * sin_t + ly * cos_t
            sx = self._world_to_screen_x(world_cx, cam_x)
            sy = self._world_to_screen_y(world_cy, cam_y)
            screen_pts.append((sx, sy))

        pygame.draw.polygon(self._screen, color, screen_pts)

        # Outline.
        outline_color = tuple(max(0, c - 40) for c in color)
        pygame.draw.polygon(self._screen, outline_color, screen_pts, 1)

        # Windshield strip (front 25% of body).
        front_frac = 0.25
        wf_local = [
            (-half_w * 0.7,  half_l),
            ( half_w * 0.7,  half_l),
            ( half_w * 0.7,  half_l - length * front_frac),
            (-half_w * 0.7,  half_l - length * front_frac),
        ]
        wf_pts = []
        for lx, ly in wf_local:
            world_cx = wx + lx * cos_t + ly * sin_t
            world_cy = wy - lx * sin_t + ly * cos_t
            wf_pts.append((
                self._world_to_screen_x(world_cx, cam_x),
                self._world_to_screen_y(world_cy, cam_y),
            ))
        glass_color = (180, 220, 240) if is_ego else (160, 190, 200)
        pygame.draw.polygon(self._screen, glass_color, wf_pts)

    # ------------------------------------------------------------------
    # Radar
    # ------------------------------------------------------------------

    def _draw_radar_cone(
        self,
        ego: Vehicle,
        cam_x: float,
        cam_y: float,
        reading: RadarReading,
    ) -> None:
        """Draw the translucent forward radar cone."""
        self._radar_surf.fill((0, 0, 0, 0))

        # Cone apex at front bumper.
        fx, fy = ego.get_front_center()
        apex_sx = self._world_to_screen_x(fx, cam_x)
        apex_sy = self._world_to_screen_y(fy, cam_y)

        fov_rad = math.radians(config.RADAR_FOV_DEG)
        range_px = config.RADAR_RANGE_M * config.PIXELS_PER_METER
        theta = ego.theta

        # Build cone polygon (apex + arc points).
        pts = [(apex_sx, apex_sy)]
        num_arc = 18
        for i in range(num_arc + 1):
            angle = theta + fov_rad * (2 * i / num_arc - 1.0)
            px = apex_sx + range_px * math.sin(angle)
            py = apex_sy - range_px * math.cos(angle)
            pts.append((px, py))

        if len(pts) >= 3:
            pygame.draw.polygon(
                self._radar_surf,
                (*config.COLOR_RADAR, config.RADAR_ALPHA),
                pts,
            )

        # Draw radar hit markers.
        for hx, hy in reading.ray_hits:
            sx = self._world_to_screen_x(hx, cam_x)
            sy = self._world_to_screen_y(hy, cam_y)
            pygame.draw.circle(
                self._radar_surf,
                (*config.COLOR_RADAR, 200),
                (int(sx), int(sy)),
                4,
            )

        self._screen.blit(self._radar_surf, (0, 0))

    # ------------------------------------------------------------------
    # Speedometer
    # ------------------------------------------------------------------

    def _draw_speedometer(self, speed_ms: float) -> None:
        """Draw a simple arc speedometer in the top-right corner."""
        cx = self._w - 80
        cy = 80
        radius = 55
        start_angle = math.radians(220)
        end_angle = math.radians(-40)   # 0 at left-bottom, max at right-bottom
        arc_span = end_angle - start_angle  # negative (CW)

        # Background arc.
        pygame.draw.circle(self._screen, config.COLOR_PANEL_BG, (cx, cy), radius)
        pygame.draw.circle(self._screen, config.COLOR_PANEL_BORDER, (cx, cy), radius, 2)

        # Speed arc.
        fraction = min(1.0, speed_ms / config.MAX_SPEED_MS)
        arc_end = start_angle + arc_span * fraction

        rect = pygame.Rect(cx - radius, cy - radius, 2 * radius, 2 * radius)
        # pygame arc angles go counter-clockwise from positive x.
        a1 = -math.degrees(start_angle)
        a2 = -math.degrees(arc_end)
        if abs(a1 - a2) > 0.5:
            arc_color = config.COLOR_WARNING if fraction > 0.85 else config.COLOR_EGO
            pygame.draw.arc(
                self._screen, arc_color, rect,
                math.radians(a2), math.radians(a1), 8
            )

        # Needle.
        needle_len = radius - 10
        angle = start_angle + arc_span * fraction
        nx = cx + int(needle_len * math.cos(angle))
        ny = cy - int(needle_len * math.sin(angle))
        pygame.draw.line(self._screen, config.COLOR_MARKING, (cx, cy), (nx, ny), 2)
        pygame.draw.circle(self._screen, config.COLOR_MARKING, (cx, cy), 4)

        # Speed text.
        font = pygame.font.SysFont("monospace", 13, bold=True)
        kph = speed_ms * 3.6
        txt = font.render(f"{kph:.0f}", True, config.COLOR_MARKING)
        self._screen.blit(txt, (cx - txt.get_width() // 2, cy + 18))
        unit = font.render("km/h", True, config.COLOR_UI_TEXT)
        self._screen.blit(unit, (cx - unit.get_width() // 2, cy + 32))

    # ------------------------------------------------------------------
    # Coordinate conversion helpers
    # ------------------------------------------------------------------

    def _world_to_screen_x(self, world_x: float, cam_x: float) -> int:
        """Convert world lateral coordinate to screen x."""
        return int(config.ROAD_CENTER_X + (world_x - cam_x) * config.PIXELS_PER_METER)

    def _world_to_screen_y(self, world_y: float, cam_y: float) -> int:
        """Convert world longitudinal coordinate to screen y.

        In screen space, +y is downward; in world space, +y is forward.
        So as world_y increases (vehicle moves forward), screen_y decreases.
        """
        return int(config.EGO_SCREEN_Y - (world_y - cam_y) * config.PIXELS_PER_METER)

    def screen_to_world(
        self, screen_x: int, screen_y: int, cam_x: float, cam_y: float
    ) -> Tuple[float, float]:
        """Convert a screen position back to world coordinates.

        Used for mapping mouse clicks to obstacle placements.

        Args:
            screen_x, screen_y: Pixel position on screen.
            cam_x, cam_y:       Current camera reference point.

        Returns:
            (world_x, world_y) in metres.
        """
        world_x = cam_x + (screen_x - config.ROAD_CENTER_X) / config.PIXELS_PER_METER
        world_y = cam_y - (screen_y - config.EGO_SCREEN_Y) / config.PIXELS_PER_METER
        return world_x, world_y
