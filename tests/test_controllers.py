"""Unit tests for ADAS controllers (PID, ACC, LKA)."""

from __future__ import annotations

import pytest

from src.controllers.pid import PIDController
from src.controllers.acc import ACC
from src.controllers.lka import LKA
from src import config


class TestPIDController:
    def test_proportional_only(self) -> None:
        """P-only controller: output = kp * error."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0)
        out = pid.update(error=3.0, dt=0.1)
        assert out == pytest.approx(6.0, abs=1e-6)

    def test_output_clamped(self) -> None:
        """Output should be clamped to [output_min, output_max]."""
        pid = PIDController(kp=10.0, ki=0.0, kd=0.0, output_min=-1.0, output_max=1.0)
        out = pid.update(error=100.0, dt=0.1)
        assert out == pytest.approx(1.0)

    def test_integral_accumulates(self) -> None:
        """Integral term should grow over time with constant error."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0, windup_limit=1000.0)
        total = 0.0
        for _ in range(10):
            total += pid.update(error=1.0, dt=0.1)
        # After 10 steps of dt=0.1 with ki=1, integral should be 1.0
        # output on last call = ki * integral = 1.0 * 1.0 = 1.0
        assert pid._integral == pytest.approx(1.0, abs=1e-9)

    def test_integral_windup_limit(self) -> None:
        """Integral should be clamped by windup_limit."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0, windup_limit=5.0)
        for _ in range(1000):
            pid.update(error=1.0, dt=0.1)
        assert abs(pid._integral) <= 5.0

    def test_derivative_on_second_call(self) -> None:
        """Derivative term: no kick on first call, present on second."""
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
        out1 = pid.update(error=0.0, dt=0.1)
        assert out1 == pytest.approx(0.0)
        out2 = pid.update(error=1.0, dt=0.1)
        # derivative = (1.0 - 0.0) / 0.1 = 10.0, kd * 10 = 10.0
        assert out2 == pytest.approx(10.0, abs=1e-6)

    def test_reset(self) -> None:
        """After reset, the controller behaves as fresh."""
        pid = PIDController(kp=1.0, ki=1.0, kd=0.0)
        for _ in range(5):
            pid.update(error=1.0, dt=0.1)
        pid.reset()
        assert pid._integral == pytest.approx(0.0)
        assert pid._prev_error == pytest.approx(0.0)


class TestACC:
    def test_accelerates_when_below_target(self) -> None:
        """ACC should command positive acceleration when speed < target."""
        acc = ACC(target_speed=30.0)
        accel = acc.compute(ego_speed=20.0, dt=0.05)
        assert accel > 0.0

    def test_decelerates_when_above_target(self) -> None:
        """ACC should command negative acceleration when speed > target."""
        acc = ACC(target_speed=20.0)
        accel = acc.compute(ego_speed=30.0, dt=0.05)
        assert accel < 0.0

    def test_brakes_near_obstacle(self) -> None:
        """ACC should command braking when a lead vehicle is dangerously close."""
        acc = ACC(target_speed=30.0)
        # Run a few steps at cruising speed to stabilise PID.
        for _ in range(5):
            acc.compute(ego_speed=30.0, dt=0.05)
        # Now a lead vehicle is within the warning threshold.
        accel = acc.compute(
            ego_speed=30.0,
            dt=0.05,
            lead_distance=config.ACC_WARNING_DISTANCE_M - 1.0,
        )
        assert accel < 0.0

    def test_output_within_limits(self) -> None:
        """ACC output must stay within physical limits."""
        acc = ACC(target_speed=30.0)
        for ego_v in [0.0, 15.0, 44.0]:
            accel = acc.compute(ego_speed=ego_v, dt=0.05)
            assert accel >= -config.MAX_DECEL_MS2
            assert accel <= config.MAX_ACCEL_MS2

    def test_reset(self) -> None:
        acc = ACC(target_speed=30.0)
        acc.compute(ego_speed=20.0, dt=0.05)
        acc.reset()
        assert acc._speed_pid._integral == pytest.approx(0.0)


class TestLKA:
    def test_corrects_right_drift(self) -> None:
        """Vehicle drifted right of lane centre → LKA should steer left (negative)."""
        lka = LKA()
        # ego_x > lane_center_x: vehicle is to the right.
        correction = lka.compute(
            ego_x=1.5, ego_theta=0.0, lane_center_x=0.0, dt=0.05
        )
        assert correction < 0.0

    def test_corrects_left_drift(self) -> None:
        """Vehicle drifted left of lane centre → LKA should steer right (positive)."""
        lka = LKA()
        correction = lka.compute(
            ego_x=-1.5, ego_theta=0.0, lane_center_x=0.0, dt=0.05
        )
        assert correction > 0.0

    def test_zero_error_zero_output(self) -> None:
        """When vehicle is centred with zero heading, correction should be ~0."""
        lka = LKA()
        correction = lka.compute(
            ego_x=0.0, ego_theta=0.0, lane_center_x=0.0, dt=0.05
        )
        assert correction == pytest.approx(0.0, abs=1e-6)

    def test_output_clamped(self) -> None:
        """LKA correction must not exceed the maximum steer correction."""
        lka = LKA()
        correction = lka.compute(
            ego_x=100.0, ego_theta=0.0, lane_center_x=0.0, dt=0.05
        )
        assert abs(correction) <= config.LKA_MAX_STEER_CORRECTION_RAD

    def test_heading_damps_correction(self) -> None:
        """A nose-right heading should reduce a rightward steer correction.

        Use a small lateral error that will not saturate the output clamp so
        the difference between the two corrections is visible.
        """
        lka_no_heading = LKA()
        lka_with_heading = LKA()
        # Both vehicles are slightly to the left (need right steer), but one
        # is already heading right (positive theta).  A small error ensures
        # neither output saturates the ±LKA_MAX_STEER_CORRECTION_RAD clamp.
        out_no_hdg = lka_no_heading.compute(-0.15, 0.0, 0.0, 0.05)
        out_with_hdg = lka_with_heading.compute(-0.15, 0.3, 0.0, 0.05)
        # Heading-right reduces the combined error, so correction is smaller.
        assert out_with_hdg < out_no_hdg
