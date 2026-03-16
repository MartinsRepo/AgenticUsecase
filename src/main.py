"""
Main entry point for the ADAS 2D highway simulation.

Architecture: strict MVC
  - Model  : src/model/vehicle.py, src/model/sensor.py
  - View   : src/view/renderer.py
  - Controller: src/controller/adas.py, src/controller/pid.py

Run with:
    python -m src.main
or
    python src/main.py
"""

from __future__ import annotations

import math
import sys
from typing import List, Optional

import pygame

from src.controller.adas import AdaptiveCruiseControl, LaneKeepingAssist
from src.model.sensor import LaneSensor, Obstacle, RadarSensor
from src.model.vehicle import Vehicle
from src.view.renderer import Renderer

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

ROAD_LEFT = 150.0
ROAD_RIGHT = 450.0
LANE_WIDTH = 100.0
NUM_LANES = 3

# Ego vehicle starts in the centre lane
EGO_START_X = ROAD_LEFT + LANE_WIDTH * 1.5
EGO_START_Y = SCREEN_HEIGHT * 0.7
EGO_START_SPEED = 120.0    # pixels / second

# NPC vehicle initial positions (x, y, speed)
NPC_CONFIGS = [
    (ROAD_LEFT + LANE_WIDTH * 0.5, SCREEN_HEIGHT * 0.3, 80.0),
    (ROAD_LEFT + LANE_WIDTH * 1.5, SCREEN_HEIGHT * 0.25, 70.0),
    (ROAD_LEFT + LANE_WIDTH * 2.5, SCREEN_HEIGHT * 0.35, 90.0),
]

# Speed adjustment step (pixels/second per key press)
SPEED_STEP = 20.0
TARGET_SPEED_MIN = 0.0
TARGET_SPEED_MAX = 300.0


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

class Simulation:
    """
    Top-level simulation coordinator.

    Owns all model objects, sensor instances, and controllers, and wires them
    together each frame.
    """

    def __init__(self) -> None:
        # --- Model ---
        self.ego = Vehicle(
            x=EGO_START_X,
            y=EGO_START_Y,
            heading=-math.pi / 2.0,
            speed=EGO_START_SPEED,
            is_ego=True,
        )

        self.npcs: List[Vehicle] = [
            Vehicle(x=x, y=y, heading=-math.pi / 2.0, speed=sp)
            for x, y, sp in NPC_CONFIGS
        ]

        self.obstacles: List[Obstacle] = []

        # --- Sensors ---
        self.radar = RadarSensor(
            max_range=300.0,
            cone_half_angle=math.radians(15.0),
            num_rays=20,
        )
        self.lane_sensor = LaneSensor(
            lane_width=LANE_WIDTH,
            road_left=ROAD_LEFT,
            num_lanes=NUM_LANES,
        )

        # --- Controllers ---
        self.acc = AdaptiveCruiseControl(target_speed=EGO_START_SPEED)
        self.lka = LaneKeepingAssist()

        # --- Feature toggles ---
        self.acc_enabled = True
        self.lka_enabled = True

        # --- Renderer (View) ---
        self.renderer = Renderer(
            screen_width=SCREEN_WIDTH,
            screen_height=SCREEN_HEIGHT,
            road_left=ROAD_LEFT,
            road_right=ROAD_RIGHT,
            lane_width=LANE_WIDTH,
            num_lanes=NUM_LANES,
            fps=FPS,
        )

        # Cached sensor readings
        self._distance_to_lead: Optional[float] = None
        self._lateral_offset: float = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Enter the main simulation loop."""
        running = True
        dt = 1.0 / FPS

        while running:
            # --- Event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_click(event.button, event.pos)

            # --- Sensor updates ---
            all_targets = self._build_sensor_targets()
            self._distance_to_lead = self.radar.update(
                origin=(self.ego.state.x, self.ego.state.y),
                heading=self.ego.heading_angle,
                obstacles=all_targets,
            )
            self._lateral_offset = self.lane_sensor.update(self.ego.state.x)

            # --- ADAS control ---
            acceleration = 0.0
            steering_correction = 0.0

            if self.acc_enabled:
                acceleration = self.acc.compute(
                    current_speed=self.ego.speed_value,
                    dt=dt,
                    distance_to_lead=self._distance_to_lead,
                )

            if self.lka_enabled:
                steering_correction = self.lka.compute(
                    lateral_offset=self._lateral_offset,
                    dt=dt,
                )

            # --- Physics update ---
            self.ego.update(dt=dt, acceleration=acceleration, steering_angle=steering_correction)
            self._update_npcs(dt)
            self._wrap_npcs()

            # --- Get actual dt from renderer tick ---
            mouse_pos = pygame.mouse.get_pos()

            self.renderer.render(
                ego=self.ego,
                npcs=self.npcs,
                obstacles=self.obstacles,
                radar=self.radar,
                distance_to_lead=self._distance_to_lead,
                lateral_offset=self._lateral_offset,
                target_speed=self.acc.target_speed,
                acc_active=self.acc_enabled,
                lka_active=self.lka_enabled,
                mouse_pos=mouse_pos,
            )

        self.renderer.quit()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _handle_key(self, key: int) -> bool:
        """
        Process keyboard input.

        Returns
        -------
        False if the simulation should quit, True otherwise.
        """
        if key == pygame.K_ESCAPE:
            return False

        if key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            new_speed = min(self.acc.target_speed + SPEED_STEP, TARGET_SPEED_MAX)
            self.acc.set_target_speed(new_speed)

        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            new_speed = max(self.acc.target_speed - SPEED_STEP, TARGET_SPEED_MIN)
            self.acc.set_target_speed(new_speed)

        elif key == pygame.K_a:
            self.acc_enabled = not self.acc_enabled

        elif key == pygame.K_l:
            self.lka_enabled = not self.lka_enabled
            self.lka.reset()

        elif key == pygame.K_r:
            self._reset()

        return True

    def _handle_mouse_click(self, button: int, pos: tuple[int, int]) -> None:
        """
        Handle mouse clicks:
        - Left click  : Place a static obstacle at the cursor position.
        - Right click : Remove the nearest obstacle within 40 px.
        """
        mx, my = pos
        if button == 1:  # Left click - place obstacle
            # Only place on the road
            if ROAD_LEFT <= mx <= ROAD_RIGHT:
                self.obstacles.append(Obstacle(x=float(mx), y=float(my)))
        elif button == 3:  # Right click - remove nearest obstacle
            self._remove_nearest_obstacle(mx, my)

    # ------------------------------------------------------------------
    # NPC logic
    # ------------------------------------------------------------------

    def _update_npcs(self, dt: float) -> None:
        """Move NPC vehicles straight ahead at constant speed."""
        for npc in self.npcs:
            npc.update(dt=dt, acceleration=0.0, steering_angle=0.0)

    def _wrap_npcs(self) -> None:
        """Wrap NPCs that have scrolled off the top of the screen back to the bottom."""
        for npc in self.npcs:
            if npc.state.y < -npc.length:
                npc.state.y = SCREEN_HEIGHT + npc.length

    # ------------------------------------------------------------------
    # Sensor target aggregation
    # ------------------------------------------------------------------

    def _build_sensor_targets(self) -> List[Obstacle]:
        """
        Combine static obstacles and NPC vehicles into a single list of
        Obstacle objects for the radar sensor to query.
        """
        targets: List[Obstacle] = list(self.obstacles)
        for npc in self.npcs:
            targets.append(
                Obstacle(
                    x=npc.state.x,
                    y=npc.state.y,
                    width=npc.width,
                    height=npc.length,
                )
            )
        return targets

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _remove_nearest_obstacle(self, mx: int, my: int) -> None:
        """Remove the static obstacle closest to (mx, my) if within 40 px."""
        if not self.obstacles:
            return
        min_dist = float("inf")
        nearest_idx = -1
        for i, obs in enumerate(self.obstacles):
            d = math.hypot(obs.x - mx, obs.y - my)
            if d < min_dist:
                min_dist = d
                nearest_idx = i
        if min_dist <= 40.0 and nearest_idx >= 0:
            self.obstacles.pop(nearest_idx)

    def _reset(self) -> None:
        """Reset the ego vehicle and clear all user-placed obstacles."""
        self.ego.state.x = EGO_START_X
        self.ego.state.y = EGO_START_Y
        self.ego.state.heading = -math.pi / 2.0
        self.ego.state.speed = EGO_START_SPEED
        self.obstacles.clear()
        self.acc.set_target_speed(EGO_START_SPEED)
        self.lka.reset()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Initialise and run the ADAS simulation."""
    sim = Simulation()
    sim.run()


if __name__ == "__main__":
    main()
