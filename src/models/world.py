"""World state – road geometry, ego vehicle, NPC traffic and obstacles.

The World is the central Model object that owns every entity in the
simulation.  The controllers and renderer read from it; only the main
loop and the World itself write to it.
"""

from __future__ import annotations

import random


from src import config
from src.models.obstacle import NPCVehicle, Obstacle, _lane_center_x
from src.models.vehicle import Vehicle


# Type alias for anything that can block the radar.
TrafficObject = NPCVehicle | Obstacle


class World:
    """Container for all simulation entities and road geometry.

    The road is centred at world x = 0.  The ego vehicle starts at the
    centre of the middle lane, heading straight (theta = 0).

    Attributes:
        ego:            The player-controlled vehicle.
        npcs:           List of NPC vehicles currently alive.
        obstacles:      List of user-placed static obstacles.
        target_speed:   ACC target speed in m/s (mutable by user input).
        lane_centers_x: World-x position of each lane centre.
        time:           Elapsed simulation time in seconds.
    """

    def __init__(self) -> None:
        # Road geometry – derived once from config.
        self.lane_centers_x: list[float] = [
            _lane_center_x(i) for i in range(config.NUM_LANES)
        ]

        # Ego vehicle starts in the middle lane, at y = 0, heading straight.
        mid = config.NUM_LANES // 2
        self.ego = Vehicle(
            x=self.lane_centers_x[mid],
            y=0.0,
            theta=0.0,
            v=config.ACC_DEFAULT_TARGET_SPEED_MS,
            color=config.COLOR_EGO,
        )

        self.npcs: list[NPCVehicle] = []
        self.obstacles: list[Obstacle] = []
        self.target_speed: float = config.ACC_DEFAULT_TARGET_SPEED_MS
        self.time: float = 0.0

        # Spawn initial NPC traffic.
        self._spawn_initial_npcs()

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def update(self, dt: float, steering: float, accel: float) -> None:
        """Advance the world by *dt* seconds.

        Args:
            dt:       Time step (seconds).
            steering: Steering angle applied to ego (rad).
            accel:    Acceleration applied to ego (m/s²).
        """
        self.time += dt
        self.ego.update(dt, steering, accel)

        for npc in self.npcs:
            npc.update(dt)

        self._manage_npc_population()

    # ------------------------------------------------------------------
    # Obstacle management
    # ------------------------------------------------------------------

    def add_obstacle(self, x: float, y: float) -> None:
        """Place a static obstacle at world position (*x*, *y*)."""
        self.obstacles.append(Obstacle(x=x, y=y))

    def clear_obstacles(self) -> None:
        """Remove all user-placed obstacles."""
        self.obstacles.clear()

    # ------------------------------------------------------------------
    # NPC population management
    # ------------------------------------------------------------------

    def _spawn_initial_npcs(self) -> None:
        """Populate the road with an initial spread of NPC vehicles."""
        lanes = list(range(config.NUM_LANES))
        random.shuffle(lanes)

        for i in range(config.NPC_COUNT):
            lane = lanes[i % len(lanes)]
            y_offset = 30.0 + i * (config.NPC_SPAWN_AHEAD_M / config.NPC_COUNT)
            self.npcs.append(NPCVehicle(lane_index=lane, y=self.ego.y + y_offset))

    def _manage_npc_population(self) -> None:
        """Cull NPCs that have fallen behind and respawn them ahead."""
        ego_y = self.ego.y
        alive: list[NPCVehicle] = []

        for npc in self.npcs:
            if npc.y > ego_y - config.NPC_CULL_BEHIND_M:
                alive.append(npc)

        # Respawn culled NPCs ahead of ego.
        while len(alive) < config.NPC_COUNT:
            lane = random.randint(0, config.NUM_LANES - 1)
            y_spawn = ego_y + config.NPC_SPAWN_AHEAD_M + random.uniform(-10.0, 10.0)
            alive.append(NPCVehicle(lane_index=lane, y=y_spawn))

        self.npcs = alive

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def all_traffic(self) -> list[TrafficObject]:
        """Return every obstacle-like entity (NPCs + static obstacles)."""
        result: list[TrafficObject] = []
        result.extend(self.npcs)
        result.extend(self.obstacles)
        return result

    def ego_lane_index(self) -> int:
        """Return the index of the lane the ego vehicle currently occupies."""
        total_width = config.NUM_LANES * config.LANE_WIDTH_M
        left_edge = -total_width / 2.0
        raw = (self.ego.x - left_edge) / config.LANE_WIDTH_M
        return int(max(0, min(config.NUM_LANES - 1, raw)))

    def ego_lane_center_x(self) -> float:
        """Return the world-x centre of the ego vehicle's current lane."""
        return self.lane_centers_x[self.ego_lane_index()]
