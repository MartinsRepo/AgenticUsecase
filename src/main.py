"""ADAS Highway Simulation – main entry point.

Architecture (MVC)
------------------
* Model      : :mod:`src.models.world` – vehicle physics, NPC traffic,
               obstacles.
* View        : :mod:`src.views.renderer` + :mod:`src.views.hud` – Pygame
               rendering pipeline.
* Controller  : :mod:`src.controllers.acc` + :mod:`src.controllers.lka` –
               ADAS closed-loop logic.

The main loop runs at exactly FPS frames per second.  On every tick:

1. Process input events (keyboard + mouse).
2. Run the ADAS controllers to compute (steering, acceleration) commands.
3. Step the World model forward by dt.
4. Render the frame via Renderer + HUD.
"""

from __future__ import annotations

import sys

import pygame

from src import config
from src.controllers.acc import ACC
from src.controllers.lka import LKA
from src.models.world import World
from src.sensors.radar import Radar, RadarReading
from src.views.hud import HUD
from src.views.renderer import Renderer


def main() -> None:
    """Initialise pygame and run the simulation loop."""
    pygame.init()
    pygame.display.set_caption(config.WINDOW_TITLE)
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    # ------------------------------------------------------------------ #
    # Model / Controller / View initialisation                             #
    # ------------------------------------------------------------------ #
    world = World()
    acc = ACC(target_speed=world.target_speed)
    lka = LKA()
    radar = Radar()
    renderer = Renderer(screen)
    hud = HUD(screen)

    # ADAS flags – always active in this MVP.
    acc_active = True
    lka_active = True

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    running = True
    while running:
        dt: float = clock.tick(config.FPS) / 1000.0   # seconds
        dt = min(dt, 0.05)                              # clamp for robustness

        # ---------------------------------------------------------------- #
        # 1. Event handling                                                 #
        # ---------------------------------------------------------------- #
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_UP:
                    world.target_speed = min(
                        config.ACC_MAX_TARGET_SPEED_MS,
                        world.target_speed + config.SPEED_INCREMENT_MS,
                    )
                    acc.target_speed = world.target_speed

                elif event.key == pygame.K_DOWN:
                    world.target_speed = max(
                        config.ACC_MIN_TARGET_SPEED_MS,
                        world.target_speed - config.SPEED_INCREMENT_MS,
                    )
                    acc.target_speed = world.target_speed

                elif event.key == pygame.K_r:
                    world.clear_obstacles()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:   # left click → place obstacle
                    mx, my = event.pos
                    wx, wy = renderer.screen_to_world(
                        mx, my, world.ego.x, world.ego.y
                    )
                    # Only place obstacles on the road surface.
                    road_half = (config.NUM_LANES * config.LANE_WIDTH_M) / 2.0
                    if abs(wx) < road_half + config.SHOULDER_WIDTH_M:
                        world.add_obstacle(wx, wy)

        # ---------------------------------------------------------------- #
        # 2. Sensor scan                                                    #
        # ---------------------------------------------------------------- #
        fx, fy = world.ego.get_front_center()
        radar_reading: RadarReading = radar.scan(
            ego_x=fx,
            ego_y=fy,
            ego_theta=world.ego.theta,
            targets=world.all_traffic(),
        )

        lead_distance = (
            radar_reading.closest.distance if radar_reading.closest else None
        )
        lead_speed = (
            radar_reading.closest.target_speed if radar_reading.closest else None
        )

        # ---------------------------------------------------------------- #
        # 3. ADAS controllers                                               #
        # ---------------------------------------------------------------- #
        accel = acc.compute(
            ego_speed=world.ego.v,
            dt=dt,
            lead_distance=lead_distance,
            lead_speed=lead_speed,
        )

        steering = lka.compute(
            ego_x=world.ego.x,
            ego_theta=world.ego.theta,
            lane_center_x=world.ego_lane_center_x(),
            dt=dt,
        )

        # ---------------------------------------------------------------- #
        # 4. World step                                                      #
        # ---------------------------------------------------------------- #
        world.update(dt, steering, accel)

        # ---------------------------------------------------------------- #
        # 5. Render                                                         #
        # ---------------------------------------------------------------- #
        warning = (
            lead_distance is not None
            and lead_distance < config.ACC_WARNING_DISTANCE_M
        )

        renderer.draw_frame(
            ego=world.ego,
            npcs=world.npcs,
            obstacles=world.obstacles,
            radar_reading=radar_reading,
            warning=warning,
        )

        hud.draw(
            ego_speed_ms=world.ego.v,
            target_speed_ms=world.target_speed,
            acc_active=acc_active,
            lka_active=lka_active,
            lead_distance=lead_distance,
            radar_reading=radar_reading,
            steering_angle=world.ego.steering_angle,
            acceleration=world.ego.acceleration,
            sim_time=world.time,
            warning=warning,
        )

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
