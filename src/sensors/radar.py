"""Radar sensor simulation using raycasting.

The radar casts a set of rays forward from the ego vehicle's front bumper
within a configurable angular field of view.  Each ray is tested against
every axis-aligned bounding box of the traffic objects in the world.  The
sensor returns the closest intersection point along with the distance and
the speed of the detected object.

Coordinate convention matches the vehicle model:
    * x – lateral (positive = right)
    * y – longitudinal (positive = forward)
    * theta = 0 means heading in the +y direction
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from src import config
from src.models.obstacle import NPCVehicle, Obstacle


# Type alias for any entity the radar can detect.
RadarTarget = Union[NPCVehicle, Obstacle]


@dataclass
class RadarHit:
    """Result of a single successful ray intersection."""

    distance: float           # metres from front bumper to hit point
    hit_x: float              # world-x of the hit point
    hit_y: float              # world-y of the hit point
    target_speed: float = 0.0 # m/s of the detected object (0 for static)


@dataclass
class RadarReading:
    """Aggregated output of one radar scan cycle."""

    # Closest detection in the forward cone (None if nothing detected).
    closest: Optional[RadarHit] = None
    # All per-ray hit points, used for visualisation.
    ray_hits: List[Tuple[float, float]] = field(default_factory=list)


class Radar:
    """Forward-looking radar sensor using raycasting.

    The sensor casts *num_rays* rays within ±fov_deg of the vehicle's
    heading.  Each ray is intersected against the axis-aligned bounding
    boxes (AABB) of all traffic objects.

    Args:
        range_m:   Maximum detection range in metres.
        fov_deg:   Half-angle of the field of view in degrees.
        num_rays:  Number of rays to cast across the FOV.
    """

    def __init__(
        self,
        range_m: float = config.RADAR_RANGE_M,
        fov_deg: float = config.RADAR_FOV_DEG,
        num_rays: int = 12,
    ) -> None:
        self.range_m = range_m
        self.fov_deg = fov_deg
        self.num_rays = num_rays

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        ego_x: float,
        ego_y: float,
        ego_theta: float,
        targets: List[RadarTarget],
    ) -> RadarReading:
        """Perform a full radar scan and return a :class:`RadarReading`.

        Args:
            ego_x:    Lateral world position of the ego front bumper (m).
            ego_y:    Longitudinal world position of the ego front bumper (m).
            ego_theta: Ego heading angle (rad); 0 = straight ahead (+y).
            targets:  List of traffic objects to test.

        Returns:
            A :class:`RadarReading` with the closest detection and all ray hits.
        """
        fov_rad = math.radians(self.fov_deg)
        angles = np.linspace(-fov_rad, fov_rad, self.num_rays)

        reading = RadarReading()
        min_dist = float("inf")

        for angle_offset in angles:
            ray_angle = ego_theta + angle_offset
            dx = math.sin(ray_angle)
            dy = math.cos(ray_angle)

            hit = self._cast_ray(ego_x, ego_y, dx, dy, targets)
            if hit is not None:
                reading.ray_hits.append((hit.hit_x, hit.hit_y))
                if hit.distance < min_dist:
                    min_dist = hit.distance
                    reading.closest = hit

        return reading

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cast_ray(
        self,
        ox: float,
        oy: float,
        dx: float,
        dy: float,
        targets: List[RadarTarget],
    ) -> Optional[RadarHit]:
        """Intersect a single ray against all target AABBs.

        Uses the slab method (Kay-Kajiya) for AABB intersection.

        Args:
            ox, oy: Ray origin in world coordinates.
            dx, dy: Unit direction vector of the ray.
            targets: List of objects to test.

        Returns:
            The closest :class:`RadarHit`, or None if no intersection
            within range.
        """
        best_t = self.range_m
        best_hit: Optional[RadarHit] = None

        for target in targets:
            half_w = target.width / 2.0
            half_l = target.length / 2.0
            tx = target.x
            ty = target.y

            t = _ray_aabb_intersect(
                ox, oy, dx, dy,
                tx - half_w, ty - half_l,
                tx + half_w, ty + half_l,
            )

            if t is not None and 0.0 < t < best_t:
                best_t = t
                speed = getattr(target, "speed", 0.0)
                best_hit = RadarHit(
                    distance=t,
                    hit_x=ox + dx * t,
                    hit_y=oy + dy * t,
                    target_speed=float(speed),
                )

        return best_hit


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _ray_aabb_intersect(
    ox: float, oy: float,
    dx: float, dy: float,
    min_x: float, min_y: float,
    max_x: float, max_y: float,
) -> Optional[float]:
    """Slab-method ray–AABB intersection test.

    Args:
        ox, oy:           Ray origin.
        dx, dy:           Ray direction (need not be normalised, but should
                          have unit magnitude for the return value to be in
                          the same units as the coordinates).
        min_x … max_y:    Axis-aligned bounding box extents.

    Returns:
        The parametric hit distance *t* such that the intersection point is
        ``(ox + dx*t, oy + dy*t)``, or ``None`` if the ray misses.
    """
    t_min = 0.0
    t_max = float("inf")

    # X slab.
    if abs(dx) < 1e-9:
        if ox < min_x or ox > max_x:
            return None
    else:
        tx1 = (min_x - ox) / dx
        tx2 = (max_x - ox) / dx
        if tx1 > tx2:
            tx1, tx2 = tx2, tx1
        t_min = max(t_min, tx1)
        t_max = min(t_max, tx2)
        if t_min > t_max:
            return None

    # Y slab.
    if abs(dy) < 1e-9:
        if oy < min_y or oy > max_y:
            return None
    else:
        ty1 = (min_y - oy) / dy
        ty2 = (max_y - oy) / dy
        if ty1 > ty2:
            ty1, ty2 = ty2, ty1
        t_min = max(t_min, ty1)
        t_max = min(t_max, ty2)
        if t_min > t_max:
            return None

    return t_min if t_min >= 0.0 else None
