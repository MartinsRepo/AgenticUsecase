"""
ADAS controller module.

Implements two driver-assistance systems:

1. AdaptiveCruiseControl (ACC)
   - Maintains a desired target speed when the road is clear.
   - Reduces speed proportionally when a leading vehicle is detected within
     the safe following distance.
   - Uses a PID controller on the speed error to output longitudinal
     acceleration commands.

2. LaneKeepingAssist (LKA)
   - Keeps the ego vehicle centred in its lane.
   - Uses a PID controller on the lateral offset to output a steering-angle
     correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.controller.pid import PIDController


# ---------------------------------------------------------------------------
# Adaptive Cruise Control
# ---------------------------------------------------------------------------

# Minimum following distance (pixels) below which full braking is applied
ACC_MIN_DISTANCE: float = 80.0
# Comfortable following distance (pixels) at which speed is not reduced
ACC_SAFE_DISTANCE: float = 200.0


@dataclass
class AdaptiveCruiseControl:
    """
    Longitudinal speed controller with gap regulation.

    Parameters
    ----------
    target_speed  : Desired cruising speed in pixels/second.
    kp, ki, kd    : PID gains for the speed controller.
    """

    target_speed: float = 150.0
    kp: float = 1.2
    ki: float = 0.15
    kd: float = 0.08

    _pid: PIDController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pid = PIDController(
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            output_min=-120.0,
            output_max=80.0,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        current_speed: float,
        dt: float,
        distance_to_lead: Optional[float],
    ) -> float:
        """
        Compute the longitudinal acceleration command.

        Parameters
        ----------
        current_speed      : Ego vehicle speed in pixels/second.
        dt                 : Time step in seconds.
        distance_to_lead   : Distance to the nearest obstacle in pixels, or
                             None if the road ahead is clear.

        Returns
        -------
        Desired acceleration in pixels/s^2.
        """
        effective_target = self._effective_target_speed(distance_to_lead)
        return self._pid.compute(
            setpoint=effective_target,
            measurement=current_speed,
            dt=dt,
        )

    def set_target_speed(self, speed: float) -> None:
        """Update the desired cruising speed and reset the integrator."""
        self.target_speed = max(0.0, speed)
        self._pid.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_target_speed(
        self, distance_to_lead: Optional[float]
    ) -> float:
        """
        Determine the speed set-point taking gap regulation into account.

        If a leading vehicle is detected:
        - Between ACC_MIN_DISTANCE and ACC_SAFE_DISTANCE: linearly interpolate
          down to zero at the minimum distance.
        - Below ACC_MIN_DISTANCE: command full stop.
        """
        if distance_to_lead is None or distance_to_lead >= ACC_SAFE_DISTANCE:
            return self.target_speed

        if distance_to_lead <= ACC_MIN_DISTANCE:
            return 0.0

        # Linear interpolation between safe and minimum distances
        ratio = (distance_to_lead - ACC_MIN_DISTANCE) / (
            ACC_SAFE_DISTANCE - ACC_MIN_DISTANCE
        )
        return self.target_speed * ratio


# ---------------------------------------------------------------------------
# Lane Keeping Assist
# ---------------------------------------------------------------------------

# Maximum steering correction that LKA may apply (radians)
LKA_MAX_CORRECTION: float = math.radians(8.0)
# Lateral offset threshold (pixels) above which LKA is considered active
LKA_ACTIVATION_THRESHOLD: float = 5.0


@dataclass
class LaneKeepingAssist:
    """
    Lateral controller that keeps the vehicle centred in its lane.

    Parameters
    ----------
    kp, ki, kd : PID gains for the lateral offset controller.
    """

    kp: float = 0.04
    ki: float = 0.001
    kd: float = 0.02

    _pid: PIDController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pid = PIDController(
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            output_min=-LKA_MAX_CORRECTION,
            output_max=LKA_MAX_CORRECTION,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(self, lateral_offset: float, dt: float) -> float:
        """
        Compute the steering-angle correction.

        Parameters
        ----------
        lateral_offset : Signed deviation from lane centre in pixels.
                         Positive = vehicle is right of centre.
        dt             : Time step in seconds.

        Returns
        -------
        Steering angle correction in radians.  A negative value steers left.
        """
        # PID: setpoint is zero (centred in lane)
        return self._pid.compute(
            setpoint=0.0,
            measurement=lateral_offset,
            dt=dt,
        )

    @property
    def is_active(self) -> bool:
        """True when LKA is currently applying a non-trivial correction."""
        return self._pid._prev_measurement is not None and abs(
            self._pid._prev_measurement or 0.0
        ) > LKA_ACTIVATION_THRESHOLD

    def reset(self) -> None:
        """Reset the internal PID state."""
        self._pid.reset()
