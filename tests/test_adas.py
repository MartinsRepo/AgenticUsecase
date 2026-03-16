"""
Tests for the ADAS controllers (ACC and LKA).
"""

import math
import pytest

from src.controller.adas import (
    AdaptiveCruiseControl,
    LaneKeepingAssist,
    ACC_SAFE_DISTANCE,
    ACC_MIN_DISTANCE,
    LKA_MAX_CORRECTION,
)


class TestAdaptiveCruiseControl:
    def test_accelerates_when_below_target_speed(self):
        acc = AdaptiveCruiseControl(target_speed=150.0)
        output = acc.compute(current_speed=100.0, dt=0.1, distance_to_lead=None)
        assert output > 0.0

    def test_decelerates_when_above_target_speed(self):
        acc = AdaptiveCruiseControl(target_speed=100.0)
        output = acc.compute(current_speed=150.0, dt=0.1, distance_to_lead=None)
        assert output < 0.0

    def test_no_obstacle_uses_full_target_speed(self):
        acc = AdaptiveCruiseControl(target_speed=150.0)
        # At exactly target speed, acceleration should be near zero
        output = acc.compute(current_speed=150.0, dt=0.1, distance_to_lead=None)
        assert abs(output) < 5.0

    def test_brakes_when_very_close(self):
        acc = AdaptiveCruiseControl(target_speed=150.0)
        # Distance below minimum should command stop (negative acceleration)
        output = acc.compute(
            current_speed=150.0, dt=0.1, distance_to_lead=ACC_MIN_DISTANCE - 10.0
        )
        assert output < 0.0

    def test_safe_distance_no_speed_reduction(self):
        acc = AdaptiveCruiseControl(target_speed=150.0)
        # At safe distance and above, effective target = full target speed
        eff = acc._effective_target_speed(ACC_SAFE_DISTANCE + 50.0)
        assert eff == pytest.approx(150.0)

    def test_minimum_distance_returns_zero_target(self):
        acc = AdaptiveCruiseControl(target_speed=150.0)
        eff = acc._effective_target_speed(ACC_MIN_DISTANCE)
        assert eff == pytest.approx(0.0, abs=1.0)

    def test_set_target_speed_clamped_positive(self):
        acc = AdaptiveCruiseControl(target_speed=100.0)
        acc.set_target_speed(-50.0)
        assert acc.target_speed == 0.0

    def test_intermediate_distance_reduces_speed(self):
        acc = AdaptiveCruiseControl(target_speed=150.0)
        mid = (ACC_MIN_DISTANCE + ACC_SAFE_DISTANCE) / 2.0
        eff = acc._effective_target_speed(mid)
        assert 0.0 < eff < 150.0


class TestLaneKeepingAssist:
    def test_right_offset_gives_negative_steering(self):
        """Vehicle right of centre should receive leftward (negative) steering."""
        lka = LaneKeepingAssist()
        correction = lka.compute(lateral_offset=20.0, dt=0.1)
        assert correction < 0.0

    def test_left_offset_gives_positive_steering(self):
        """Vehicle left of centre should receive rightward (positive) steering."""
        lka = LaneKeepingAssist()
        correction = lka.compute(lateral_offset=-20.0, dt=0.1)
        assert correction > 0.0

    def test_centred_no_correction(self):
        lka = LaneKeepingAssist()
        correction = lka.compute(lateral_offset=0.0, dt=0.1)
        assert abs(correction) < 1e-9

    def test_correction_bounded(self):
        lka = LaneKeepingAssist()
        correction = lka.compute(lateral_offset=1000.0, dt=0.1)
        assert abs(correction) <= LKA_MAX_CORRECTION + 1e-9

    def test_reset_clears_state(self):
        lka = LaneKeepingAssist()
        for _ in range(20):
            lka.compute(lateral_offset=50.0, dt=0.1)
        lka.reset()
        # After reset, integral is cleared so small offset gives small correction
        correction = lka.compute(lateral_offset=1.0, dt=0.1)
        assert abs(correction) < 0.1
