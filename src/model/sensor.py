"""
Sensor simulation module.

Provides two sensor types required by the ADAS system:

1. RadarSensor  - forward-facing distance sensor using ray-casting to detect
                  the nearest obstacle in the ego vehicle's path.
2. LaneSensor   - lateral lane-offset sensor that measures how far the vehicle
                  centre is from the lane centre line.

All geometry is performed in world-pixel coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Obstacle representation
# ---------------------------------------------------------------------------

@dataclass
class Obstacle:
    """
    A rectangular obstacle on the road.

    Attributes
    ----------
    x, y    : World centre position (pixels).
    width   : Width in pixels (road-axis).
    height  : Height in pixels (travel-axis).
    """

    x: float
    y: float
    width: float = 40.0
    height: float = 60.0

    def rect_corners(self) -> List[Tuple[float, float]]:
        """Return axis-aligned bounding box corners [(x0,y0),(x1,y1)]."""
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        return [
            (self.x - half_w, self.y - half_h),
            (self.x + half_w, self.y + half_h),
        ]


# ---------------------------------------------------------------------------
# Forward radar sensor
# ---------------------------------------------------------------------------

@dataclass
class RadarSensor:
    """
    Simulates a forward-looking radar/lidar sensor using ray-casting.

    The sensor casts a number of rays in a forward cone and returns the
    distance to the nearest detected obstacle.

    Parameters
    ----------
    max_range       : Maximum detection range in pixels.
    cone_half_angle : Half-width of the detection cone in radians.
    num_rays        : Number of rays distributed across the cone.
    """

    max_range: float = 300.0
    cone_half_angle: float = math.radians(15.0)
    num_rays: int = 16

    # Most recent reading (pixels); None means no obstacle in range
    distance: Optional[float] = field(default=None, init=False)
    # Hit point for visualisation (None if no hit)
    hit_point: Optional[Tuple[float, float]] = field(default=None, init=False)

    def update(
        self,
        origin: Tuple[float, float],
        heading: float,
        obstacles: List[Obstacle],
    ) -> Optional[float]:
        """
        Cast rays and update the measured distance.

        Parameters
        ----------
        origin    : Sensor origin in world coordinates (front of ego vehicle).
        heading   : Heading of the ego vehicle in radians.
        obstacles : List of obstacles to check against.

        Returns
        -------
        Minimum detected distance, or None if nothing is in range.
        """
        min_dist: Optional[float] = None
        best_hit: Optional[Tuple[float, float]] = None

        angles = np.linspace(
            heading - self.cone_half_angle,
            heading + self.cone_half_angle,
            self.num_rays,
        )

        for angle in angles:
            hit, dist = self._cast_ray(origin, angle, obstacles)
            if hit is not None and (min_dist is None or dist < min_dist):  # type: ignore[operator]
                min_dist = dist
                best_hit = hit

        self.distance = min_dist
        self.hit_point = best_hit
        return min_dist

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cast_ray(
        self,
        origin: Tuple[float, float],
        angle: float,
        obstacles: List[Obstacle],
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """
        Cast a single ray and return (hit_point, distance) or (None, max_range).

        The ray is tested against each obstacle's axis-aligned bounding box
        using the slab method (Liang-Barsky algorithm variant).
        """
        ox, oy = origin
        dx = math.cos(angle)
        dy = math.sin(angle)

        closest: Optional[Tuple[Tuple[float, float], float]] = None

        for obs in obstacles:
            corners = obs.rect_corners()
            x_min = corners[0][0]
            y_min = corners[0][1]
            x_max = corners[1][0]
            y_max = corners[1][1]

            t_enter, t_exit = 0.0, self.max_range

            for axis_min, axis_max, d, o in [
                (x_min, x_max, dx, ox),
                (y_min, y_max, dy, oy),
            ]:
                if abs(d) < 1e-9:
                    if o < axis_min or o > axis_max:
                        t_enter = float("inf")
                    continue
                t1 = (axis_min - o) / d
                t2 = (axis_max - o) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_enter = max(t_enter, t1)
                t_exit = min(t_exit, t2)

            if t_enter <= t_exit and t_enter < self.max_range and t_enter >= 0:
                hit = (ox + dx * t_enter, oy + dy * t_enter)
                if closest is None or t_enter < closest[1]:
                    closest = (hit, t_enter)

        if closest is not None:
            return closest[0], closest[1]
        return None, self.max_range


# ---------------------------------------------------------------------------
# Lane sensor
# ---------------------------------------------------------------------------

@dataclass
class LaneSensor:
    """
    Measures the lateral offset of the vehicle from the centre of its lane.

    Positive offset means the vehicle is to the right of the lane centre;
    negative means it is to the left (using screen coordinates where Y
    increases downward).

    Parameters
    ----------
    lane_width : Width of a single lane in pixels.
    road_left  : X coordinate of the leftmost road boundary (pixels).
    num_lanes  : Total number of lanes on the road.
    """

    lane_width: float = 100.0
    road_left: float = 150.0
    num_lanes: int = 3

    # Latest reading
    lateral_offset: float = field(default=0.0, init=False)
    current_lane: int = field(default=1, init=False)

    def update(self, vehicle_x: float) -> float:
        """
        Compute and store the lateral offset.

        Parameters
        ----------
        vehicle_x : The X-axis world position of the vehicle centre.

        Returns
        -------
        Lateral offset from the nearest lane centre in pixels.
        """
        # Determine which lane the vehicle occupies
        relative_x = vehicle_x - self.road_left
        lane_index = int(relative_x / self.lane_width)
        lane_index = max(0, min(lane_index, self.num_lanes - 1))

        lane_centre = self.road_left + (lane_index + 0.5) * self.lane_width
        self.lateral_offset = vehicle_x - lane_centre
        self.current_lane = lane_index
        return self.lateral_offset
