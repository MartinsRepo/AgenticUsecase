"""Unit tests for the radar sensor (raycasting)."""

from __future__ import annotations

import math
import pytest

from src.sensors.radar import Radar, _ray_aabb_intersect
from src.models.obstacle import Obstacle, NPCVehicle


class TestRayAABBIntersect:
    def test_direct_hit(self) -> None:
        """Ray pointing directly into an AABB should return a positive t."""
        t = _ray_aabb_intersect(
            ox=0.0, oy=0.0, dx=0.0, dy=1.0,
            min_x=-1.0, min_y=5.0, max_x=1.0, max_y=8.0,
        )
        assert t is not None
        assert t == pytest.approx(5.0, abs=1e-6)

    def test_miss_sideways(self) -> None:
        """Ray parallel to box but displaced should miss."""
        t = _ray_aabb_intersect(
            ox=5.0, oy=0.0, dx=0.0, dy=1.0,
            min_x=-1.0, min_y=5.0, max_x=1.0, max_y=8.0,
        )
        assert t is None

    def test_ray_origin_inside_box(self) -> None:
        """Ray originating inside the box should return a non-negative t."""
        t = _ray_aabb_intersect(
            ox=0.0, oy=6.0, dx=0.0, dy=1.0,
            min_x=-1.0, min_y=5.0, max_x=1.0, max_y=8.0,
        )
        # t_min = 0 (inside), intersection happens immediately.
        assert t is not None
        assert t >= 0.0

    def test_ray_behind_box(self) -> None:
        """Ray pointing away from the box should return None."""
        t = _ray_aabb_intersect(
            ox=0.0, oy=0.0, dx=0.0, dy=-1.0,
            min_x=-1.0, min_y=5.0, max_x=1.0, max_y=8.0,
        )
        assert t is None


class TestRadarScan:
    def _make_obstacle_ahead(self, distance: float) -> Obstacle:
        """Create a static obstacle *distance* metres ahead (in +y)."""
        return Obstacle(x=0.0, y=distance, width=2.0, length=4.5)

    def test_detects_obstacle_ahead(self) -> None:
        """Radar should detect an obstacle directly ahead."""
        radar = Radar(range_m=80.0, fov_deg=15.0, num_rays=9)
        obs = self._make_obstacle_ahead(30.0)
        reading = radar.scan(ego_x=0.0, ego_y=0.0, ego_theta=0.0, targets=[obs])
        assert reading.closest is not None
        assert reading.closest.distance == pytest.approx(30.0 - 4.5 / 2, abs=0.5)

    def test_no_detection_out_of_range(self) -> None:
        """Obstacle beyond radar range should not be detected."""
        radar = Radar(range_m=20.0, fov_deg=15.0, num_rays=9)
        obs = self._make_obstacle_ahead(50.0)
        reading = radar.scan(ego_x=0.0, ego_y=0.0, ego_theta=0.0, targets=[obs])
        assert reading.closest is None

    def test_no_detection_behind(self) -> None:
        """Obstacle directly behind should not be detected by the forward radar."""
        radar = Radar(range_m=80.0, fov_deg=15.0, num_rays=9)
        obs = Obstacle(x=0.0, y=-30.0, width=2.0, length=4.5)
        reading = radar.scan(ego_x=0.0, ego_y=0.0, ego_theta=0.0, targets=[obs])
        assert reading.closest is None

    def test_closest_of_two(self) -> None:
        """Radar should return the closer of two obstacles ahead."""
        radar = Radar(range_m=80.0, fov_deg=15.0, num_rays=9)
        near = self._make_obstacle_ahead(20.0)
        far = self._make_obstacle_ahead(60.0)
        reading = radar.scan(ego_x=0.0, ego_y=0.0, ego_theta=0.0, targets=[near, far])
        assert reading.closest is not None
        assert reading.closest.distance < 25.0

    def test_miss_outside_fov(self) -> None:
        """Obstacle well outside the horizontal FOV should not be detected."""
        radar = Radar(range_m=80.0, fov_deg=10.0, num_rays=5)
        # Obstacle 50m ahead but 30m to the right of the lane.
        obs = Obstacle(x=30.0, y=50.0, width=2.0, length=4.5)
        reading = radar.scan(ego_x=0.0, ego_y=0.0, ego_theta=0.0, targets=[obs])
        assert reading.closest is None

    def test_npc_vehicle_detected(self) -> None:
        """NPC vehicles should also be detectable."""
        radar = Radar(range_m=80.0, fov_deg=15.0, num_rays=9)
        npc = NPCVehicle(lane_index=1, y=25.0, speed=20.0)
        # Place NPC directly ahead by resetting its x to 0.
        npc.x = 0.0
        reading = radar.scan(ego_x=0.0, ego_y=0.0, ego_theta=0.0, targets=[npc])
        assert reading.closest is not None
        assert reading.closest.target_speed == pytest.approx(20.0)
