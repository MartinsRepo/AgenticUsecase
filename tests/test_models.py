"""Unit tests for vehicle physics (bicycle model)."""

from __future__ import annotations

import math
import pytest

from src.models.vehicle import Vehicle, VehicleState


class TestVehicleState:
    def test_to_array(self) -> None:
        state = VehicleState(x=1.0, y=2.0, theta=0.5, v=10.0)
        arr = state.to_array()
        assert arr[0] == pytest.approx(1.0)
        assert arr[1] == pytest.approx(2.0)
        assert arr[2] == pytest.approx(0.5)
        assert arr[3] == pytest.approx(10.0)


class TestVehicleUpdate:
    def test_straight_ahead(self) -> None:
        """Vehicle heading theta=0 should move purely in +y at speed v."""
        v = Vehicle(x=0.0, y=0.0, theta=0.0, v=10.0)
        v.update(dt=1.0, steering_angle=0.0, acceleration=0.0)
        assert v.x == pytest.approx(0.0, abs=1e-6)
        assert v.y == pytest.approx(10.0, abs=1e-4)
        assert v.theta == pytest.approx(0.0, abs=1e-6)
        assert v.v == pytest.approx(10.0, abs=1e-6)

    def test_acceleration(self) -> None:
        """Positive acceleration increases speed."""
        v = Vehicle(x=0.0, y=0.0, theta=0.0, v=0.0)
        v.update(dt=1.0, steering_angle=0.0, acceleration=2.0)
        assert v.v == pytest.approx(2.0, abs=1e-6)

    def test_deceleration_clamps_at_zero(self) -> None:
        """Vehicle speed should not go below zero."""
        v = Vehicle(x=0.0, y=0.0, theta=0.0, v=5.0)
        v.update(dt=2.0, steering_angle=0.0, acceleration=-10.0)
        assert v.v >= 0.0

    def test_speed_clamped_at_max(self) -> None:
        """Vehicle speed should not exceed max_speed."""
        v = Vehicle(x=0.0, y=0.0, theta=0.0, v=48.0, max_speed=50.0)
        v.update(dt=10.0, steering_angle=0.0, acceleration=10.0)
        assert v.v <= 50.0

    def test_steering_changes_heading(self) -> None:
        """Non-zero steering should change the heading angle."""
        v = Vehicle(x=0.0, y=0.0, theta=0.0, v=10.0)
        v.update(dt=0.5, steering_angle=0.2, acceleration=0.0)
        assert v.theta != pytest.approx(0.0, abs=1e-6)

    def test_steering_clamped(self) -> None:
        """Steering input should be clamped to max_steer."""
        v = Vehicle(max_steer=0.3)
        v.update(dt=1.0, steering_angle=2.0, acceleration=0.0)
        assert abs(v.steering_angle) <= 0.3

    def test_heading_normalised(self) -> None:
        """Heading angle should remain in (-pi, pi]."""
        v = Vehicle(x=0.0, y=0.0, theta=3.0, v=5.0)
        for _ in range(100):
            v.update(dt=0.05, steering_angle=0.3, acceleration=0.0)
        assert -math.pi < v.theta <= math.pi

    def test_lateral_motion(self) -> None:
        """Vehicle heading theta=pi/2 should move purely in +x direction."""
        v = Vehicle(x=0.0, y=0.0, theta=math.pi / 2, v=10.0)
        v.update(dt=1.0, steering_angle=0.0, acceleration=0.0)
        assert v.x == pytest.approx(10.0, abs=1e-4)
        assert v.y == pytest.approx(0.0, abs=1e-4)


class TestVehicleCorners:
    def test_corners_count(self) -> None:
        v = Vehicle()
        assert len(v.get_corners()) == 4

    def test_front_center(self) -> None:
        """Front bumper should be half_length ahead in the heading direction."""
        v = Vehicle(x=0.0, y=0.0, theta=0.0, v=0.0, length=4.0)
        fx, fy = v.get_front_center()
        assert fx == pytest.approx(0.0, abs=1e-6)
        assert fy == pytest.approx(2.0, abs=1e-6)

    def test_corners_symmetric(self) -> None:
        """For theta=0, left and right corners are symmetric about x=0."""
        v = Vehicle(x=0.0, y=0.0, theta=0.0, v=0.0, width=2.0, length=4.0)
        corners = v.get_corners()
        # front-left and front-right x values should be ±1.0
        assert corners[0][0] == pytest.approx(-1.0, abs=1e-6)
        assert corners[1][0] == pytest.approx(1.0, abs=1e-6)
