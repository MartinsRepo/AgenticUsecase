"""
Tests for the vehicle bicycle model.
"""

import math
import pytest

from src.model.vehicle import Vehicle, VehicleState, WHEELBASE, MAX_SPEED


class TestVehicleState:
    def test_position(self):
        state = VehicleState(x=10.0, y=20.0)
        assert state.position() == (10.0, 20.0)

    def test_front_axle(self):
        state = VehicleState(x=0.0, y=0.0, heading=0.0)
        fx, fy = state.front_axle(wheelbase=40.0)
        assert abs(fx - 40.0) < 1e-6
        assert abs(fy - 0.0) < 1e-6


class TestVehicle:
    def test_initial_state(self):
        v = Vehicle(x=100.0, y=200.0, speed=50.0)
        assert v.state.x == 100.0
        assert v.state.y == 200.0
        assert v.state.speed == 50.0

    def test_forward_motion(self):
        """Vehicle with zero steering should move along its heading direction."""
        v = Vehicle(x=0.0, y=0.0, heading=0.0, speed=100.0)
        v.update(dt=1.0, acceleration=0.0, steering_angle=0.0)
        assert abs(v.state.x - 100.0) < 1.0
        assert abs(v.state.y) < 1.0

    def test_acceleration(self):
        v = Vehicle(x=0.0, y=0.0, heading=0.0, speed=0.0)
        v.update(dt=1.0, acceleration=50.0, steering_angle=0.0)
        assert v.state.speed == pytest.approx(50.0, abs=1.0)

    def test_speed_clamped_to_max(self):
        v = Vehicle(x=0.0, y=0.0, speed=MAX_SPEED - 1.0)
        v.update(dt=1.0, acceleration=200.0, steering_angle=0.0)
        assert v.state.speed <= MAX_SPEED

    def test_speed_cannot_go_negative(self):
        v = Vehicle(x=0.0, y=0.0, speed=10.0)
        v.update(dt=1.0, acceleration=-200.0, steering_angle=0.0)
        assert v.state.speed >= 0.0

    def test_steering_changes_heading(self):
        v = Vehicle(x=0.0, y=0.0, heading=0.0, speed=100.0)
        initial_heading = v.state.heading
        v.update(dt=0.1, acceleration=0.0, steering_angle=0.2)
        assert v.state.heading != initial_heading

    def test_corners_returns_four_points(self):
        v = Vehicle(x=100.0, y=100.0, heading=0.0)
        corners = v.corners()
        assert len(corners) == 4

    def test_speed_kmh(self):
        v = Vehicle(speed=100.0)
        # 100 px/s * 0.1 m/px * 3.6 = 36 km/h
        assert v.speed_kmh() == pytest.approx(36.0, abs=0.1)

    def test_heading_normalised(self):
        """Heading should stay in [-pi, pi] after large steering input."""
        v = Vehicle(x=0.0, y=0.0, heading=0.0, speed=200.0)
        for _ in range(100):
            v.update(dt=0.1, acceleration=0.0, steering_angle=0.5)
        assert -math.pi <= v.state.heading <= math.pi
