"""
Tests for the sensor simulation module.
"""

import math
import pytest

from src.model.sensor import Obstacle, RadarSensor, LaneSensor


class TestObstacle:
    def test_rect_corners(self):
        obs = Obstacle(x=100.0, y=100.0, width=40.0, height=60.0)
        corners = obs.rect_corners()
        assert corners[0] == (80.0, 70.0)
        assert corners[1] == (120.0, 130.0)


class TestRadarSensor:
    def test_no_obstacles_returns_none(self):
        sensor = RadarSensor(max_range=200.0)
        dist = sensor.update(origin=(0.0, 0.0), heading=0.0, obstacles=[])
        assert dist is None

    def test_detects_obstacle_directly_ahead(self):
        sensor = RadarSensor(max_range=300.0, cone_half_angle=math.radians(15.0))
        # Obstacle directly to the right at x=100 (heading=0)
        obs = Obstacle(x=150.0, y=0.0, width=40.0, height=40.0)
        dist = sensor.update(origin=(0.0, 0.0), heading=0.0, obstacles=[obs])
        assert dist is not None
        assert dist < 300.0
        assert dist > 0.0

    def test_obstacle_behind_not_detected(self):
        sensor = RadarSensor(max_range=300.0, cone_half_angle=math.radians(15.0))
        # Obstacle behind the sensor
        obs = Obstacle(x=-200.0, y=0.0, width=40.0, height=40.0)
        dist = sensor.update(origin=(0.0, 0.0), heading=0.0, obstacles=[obs])
        assert dist is None

    def test_obstacle_beyond_max_range_not_detected(self):
        sensor = RadarSensor(max_range=100.0)
        obs = Obstacle(x=500.0, y=0.0, width=40.0, height=40.0)
        dist = sensor.update(origin=(0.0, 0.0), heading=0.0, obstacles=[obs])
        assert dist is None

    def test_nearest_obstacle_returned(self):
        sensor = RadarSensor(max_range=600.0, cone_half_angle=math.radians(15.0))
        obs_near = Obstacle(x=100.0, y=0.0, width=40.0, height=40.0)
        obs_far = Obstacle(x=400.0, y=0.0, width=40.0, height=40.0)
        dist = sensor.update(origin=(0.0, 0.0), heading=0.0, obstacles=[obs_near, obs_far])
        assert dist is not None
        assert dist < 200.0


class TestLaneSensor:
    def test_centred_offset_is_near_zero(self):
        sensor = LaneSensor(lane_width=100.0, road_left=150.0, num_lanes=3)
        # Centre of lane 1 = 150 + 1.5 * 100 = 300
        offset = sensor.update(vehicle_x=300.0)
        assert abs(offset) < 1.0

    def test_right_of_centre_positive_offset(self):
        sensor = LaneSensor(lane_width=100.0, road_left=150.0, num_lanes=3)
        # Centre of lane 1 at x=300; vehicle at x=320
        offset = sensor.update(vehicle_x=320.0)
        assert offset > 0.0

    def test_left_of_centre_negative_offset(self):
        sensor = LaneSensor(lane_width=100.0, road_left=150.0, num_lanes=3)
        # Centre of lane 1 at x=300; vehicle at x=280
        offset = sensor.update(vehicle_x=280.0)
        assert offset < 0.0

    def test_lane_index_clamped(self):
        sensor = LaneSensor(lane_width=100.0, road_left=150.0, num_lanes=3)
        # Vehicle far to the right - should clamp to last lane
        sensor.update(vehicle_x=1000.0)
        assert sensor.current_lane == 2

    def test_lane_index_clamped_left(self):
        sensor = LaneSensor(lane_width=100.0, road_left=150.0, num_lanes=3)
        sensor.update(vehicle_x=0.0)
        assert sensor.current_lane == 0
