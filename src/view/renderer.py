"""
Pygame rendering layer (View component of MVC).

Responsible for all drawing operations:
  - Road surface and lane markings
  - Ego vehicle and NPC traffic
  - Radar sensor cone with transparency
  - Warning indicators (proximity alert)
  - HUD / telemetry overlay
  - Obstacle placement preview on mouse hover

All colours match the project specification:
  Asphalt    #2c3e50   Road background
  Markings   #ffffff   Lane lines
  Ego        #209dd7   Player vehicle
  Sensor     #ecad0a   Radar cone (accent yellow)
  Warning    #e74c3c   Proximity alert
  UI text    #888888   HUD telemetry
  NPC        #27ae60   Traffic vehicles
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import pygame

from src.model.sensor import Obstacle, RadarSensor
from src.model.vehicle import Vehicle

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COLOR_ASPHALT = (44, 62, 80)       # #2c3e50
COLOR_MARKINGS = (255, 255, 255)   # #ffffff
COLOR_EGO = (32, 157, 215)         # #209dd7
COLOR_SENSOR = (236, 173, 10)      # #ecad0a
COLOR_WARNING = (231, 76, 60)      # #e74c3c
COLOR_UI_TEXT = (136, 136, 136)    # #888888
COLOR_NPC = (39, 174, 96)          # #27ae60
COLOR_OBSTACLE = (192, 57, 43)     # #c0392b (static obstacle)
COLOR_ROAD_EDGE = (255, 255, 255)  # solid white edge lines

# Radar cone transparency (0-255)
SENSOR_ALPHA = 60

# Dashed marking parameters
DASH_LENGTH = 40
DASH_GAP = 30


class Renderer:
    """
    Manages the Pygame window and all draw calls.

    Parameters
    ----------
    screen_width  : Window width in pixels.
    screen_height : Window height in pixels.
    road_left     : X coordinate of the leftmost lane boundary.
    road_right    : X coordinate of the rightmost lane boundary.
    lane_width    : Width of a single lane in pixels.
    num_lanes     : Total number of traffic lanes.
    fps           : Target frame rate.
    """

    def __init__(
        self,
        screen_width: int = 800,
        screen_height: int = 600,
        road_left: float = 150.0,
        road_right: float = 450.0,
        lane_width: float = 100.0,
        num_lanes: int = 3,
        fps: int = 60,
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.road_left = road_left
        self.road_right = road_right
        self.lane_width = lane_width
        self.num_lanes = num_lanes
        self.fps = fps

        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("ADAS Simulation - LKA & ACC")
        self.clock = pygame.time.Clock()

        # Font for HUD
        self.font_large = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 13)

        # Scrolling offset for road markings (pixels, increases over time)
        self._road_offset: float = 0.0

    # ------------------------------------------------------------------
    # Main render entry point
    # ------------------------------------------------------------------

    def render(
        self,
        ego: Vehicle,
        npcs: List[Vehicle],
        obstacles: List[Obstacle],
        radar: RadarSensor,
        distance_to_lead: Optional[float],
        lateral_offset: float,
        target_speed: float,
        acc_active: bool,
        lka_active: bool,
        mouse_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Render one frame.

        Parameters
        ----------
        ego               : The ego vehicle to draw.
        npcs              : List of NPC vehicles to draw.
        obstacles         : List of static obstacles to draw.
        radar             : RadarSensor instance (for cone visualisation).
        distance_to_lead  : Latest radar distance reading.
        lateral_offset    : Latest lateral offset from LaneSensor.
        target_speed      : ACC target speed in pixels/second.
        acc_active        : Whether ACC is enabled.
        lka_active        : Whether LKA is enabled.
        mouse_pos         : Current mouse position for obstacle preview.
        """
        # Advance road scroll by ego speed (simulates forward motion)
        self._road_offset = (
            self._road_offset + ego.speed_value * (1.0 / self.fps)
        ) % (DASH_LENGTH + DASH_GAP)

        self._draw_road()
        self._draw_obstacles(obstacles)
        self._draw_npcs(npcs)
        self._draw_radar_cone(ego, radar, distance_to_lead)
        self._draw_ego(ego, distance_to_lead)

        if mouse_pos is not None:
            self._draw_obstacle_preview(mouse_pos)

        self._draw_hud(
            ego=ego,
            distance_to_lead=distance_to_lead,
            lateral_offset=lateral_offset,
            target_speed=target_speed,
            acc_active=acc_active,
            lka_active=lka_active,
        )

        pygame.display.flip()
        self.clock.tick(self.fps)

    # ------------------------------------------------------------------
    # Road
    # ------------------------------------------------------------------

    def _draw_road(self) -> None:
        """Draw asphalt background, edge lines, and dashed lane dividers."""
        # Asphalt background
        self.screen.fill(COLOR_ASPHALT)

        road_l = int(self.road_left)
        road_r = int(self.road_right)

        # Solid edge lines
        pygame.draw.line(self.screen, COLOR_ROAD_EDGE, (road_l, 0), (road_l, self.screen_height), 3)
        pygame.draw.line(self.screen, COLOR_ROAD_EDGE, (road_r, 0), (road_r, self.screen_height), 3)

        # Dashed internal lane dividers
        for lane in range(1, self.num_lanes):
            x = int(self.road_left + lane * self.lane_width)
            y = -int(self._road_offset)
            while y < self.screen_height:
                pygame.draw.line(
                    self.screen,
                    COLOR_MARKINGS,
                    (x, y),
                    (x, min(y + DASH_LENGTH, self.screen_height)),
                    2,
                )
                y += DASH_LENGTH + DASH_GAP

    # ------------------------------------------------------------------
    # Vehicles
    # ------------------------------------------------------------------

    def _draw_vehicle(
        self, vehicle: Vehicle, color: Tuple[int, int, int]
    ) -> None:
        """Draw a vehicle as a rotated rectangle with rounded corners."""
        corners = [(int(cx), int(cy)) for cx, cy in vehicle.corners()]
        pygame.draw.polygon(self.screen, color, corners)
        # Thin dark outline
        pygame.draw.polygon(self.screen, (20, 20, 20), corners, 2)

        # Heading indicator (small line at front centre)
        cx = sum(c[0] for c in corners[:2]) // 2
        cy = sum(c[1] for c in corners[:2]) // 2
        pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), 3)

    def _draw_ego(
        self, ego: Vehicle, distance_to_lead: Optional[float]
    ) -> None:
        """Draw the ego vehicle, switching to warning colour when too close."""
        too_close = (
            distance_to_lead is not None and distance_to_lead < 100.0
        )
        color = COLOR_WARNING if too_close else COLOR_EGO
        self._draw_vehicle(ego, color)

    def _draw_npcs(self, npcs: List[Vehicle]) -> None:
        """Draw all NPC vehicles."""
        for npc in npcs:
            self._draw_vehicle(npc, COLOR_NPC)

    def _draw_obstacles(self, obstacles: List[Obstacle]) -> None:
        """Draw static rectangular obstacles."""
        for obs in obstacles:
            rect = pygame.Rect(
                int(obs.x - obs.width / 2),
                int(obs.y - obs.height / 2),
                int(obs.width),
                int(obs.height),
            )
            pygame.draw.rect(self.screen, COLOR_OBSTACLE, rect)
            pygame.draw.rect(self.screen, (80, 20, 10), rect, 2)

    # ------------------------------------------------------------------
    # Radar cone
    # ------------------------------------------------------------------

    def _draw_radar_cone(
        self,
        ego: Vehicle,
        radar: RadarSensor,
        distance_to_lead: Optional[float],
    ) -> None:
        """
        Draw a semi-transparent sensor cone in front of the ego vehicle.

        The cone is drawn on a temporary surface with an alpha channel to
        achieve transparency.
        """
        origin_x, origin_y = ego.state.x, ego.state.y
        heading = ego.heading_angle
        half_angle = radar.cone_half_angle
        cone_length = radar.max_range

        # Build polygon points for the cone
        cone_surface = pygame.Surface(
            (self.screen_width, self.screen_height), pygame.SRCALPHA
        )

        num_pts = 20
        points = [(origin_x, origin_y)]
        for i in range(num_pts + 1):
            a = (heading - half_angle) + i * (2 * half_angle / num_pts)
            px = origin_x + cone_length * math.cos(a)
            py = origin_y + cone_length * math.sin(a)
            points.append((px, py))

        cone_color = (*COLOR_SENSOR, SENSOR_ALPHA)
        pygame.draw.polygon(cone_surface, cone_color, points)

        # If a hit was detected, draw a bright point and distance arc
        if radar.hit_point is not None:
            hit_color = (*COLOR_WARNING, 200)
            pygame.draw.circle(
                cone_surface,
                hit_color,
                (int(radar.hit_point[0]), int(radar.hit_point[1])),
                6,
            )

        self.screen.blit(cone_surface, (0, 0))

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(
        self,
        ego: Vehicle,
        distance_to_lead: Optional[float],
        lateral_offset: float,
        target_speed: float,
        acc_active: bool,
        lka_active: bool,
    ) -> None:
        """Render telemetry data on the left and right margins."""
        # Left panel background
        panel = pygame.Surface((140, self.screen_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 120))
        self.screen.blit(panel, (0, 0))

        # Right panel background
        r_panel = pygame.Surface((160, self.screen_height), pygame.SRCALPHA)
        r_panel.fill((0, 0, 0, 120))
        self.screen.blit(r_panel, (self.screen_width - 160, 0))

        lines_left = [
            "ADAS SIMULATION",
            "",
            f"Speed:  {ego.speed_kmh():5.1f} km/h",
            f"Target: {target_speed * 0.1 * 3.6:5.1f} km/h",
            "",
            f"Dist:   {distance_to_lead:.0f} px"
            if distance_to_lead is not None
            else "Dist:    ---  px",
            f"Offset: {lateral_offset:+.1f} px",
            "",
            f"ACC: {'ON ' if acc_active else 'OFF'}",
            f"LKA: {'ON ' if lka_active else 'OFF'}",
        ]

        lines_right = [
            "CONTROLS",
            "",
            "Mouse L: Obstacle",
            "Mouse R: Remove",
            "",
            "+ / -: Speed",
            "A: Toggle ACC",
            "L: Toggle LKA",
            "R: Reset",
            "",
            "ESC: Quit",
        ]

        y = 10
        for line in lines_left:
            color = COLOR_EGO if line.startswith("ADAS") else COLOR_UI_TEXT
            surf = self.font_small.render(line, True, color)
            self.screen.blit(surf, (8, y))
            y += 20

        y = 10
        for line in lines_right:
            color = COLOR_EGO if line.startswith("CONTROLS") else COLOR_UI_TEXT
            surf = self.font_small.render(line, True, color)
            self.screen.blit(surf, (self.screen_width - 155, y))
            y += 20

        # Warning banner if proximity alert
        if distance_to_lead is not None and distance_to_lead < 100.0:
            banner = self.font_large.render(
                "  COLLISION WARNING  ", True, COLOR_ASPHALT
            )
            banner_rect = banner.get_rect(
                center=(self.screen_width // 2, 30)
            )
            pygame.draw.rect(
                self.screen, COLOR_WARNING, banner_rect.inflate(12, 6), border_radius=4
            )
            self.screen.blit(banner, banner_rect)

    # ------------------------------------------------------------------
    # Mouse hover preview
    # ------------------------------------------------------------------

    def _draw_obstacle_preview(self, mouse_pos: Tuple[int, int]) -> None:
        """Draw a translucent obstacle rectangle at the current mouse cursor."""
        obs_w, obs_h = 40, 60
        preview = pygame.Surface((obs_w, obs_h), pygame.SRCALPHA)
        preview.fill((*COLOR_OBSTACLE, 100))
        mx, my = mouse_pos
        self.screen.blit(preview, (mx - obs_w // 2, my - obs_h // 2))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def tick(self) -> float:
        """Advance the clock and return the actual delta time in seconds."""
        return self.clock.tick(self.fps) / 1000.0

    def quit(self) -> None:
        """Cleanly shut down Pygame."""
        pygame.quit()
