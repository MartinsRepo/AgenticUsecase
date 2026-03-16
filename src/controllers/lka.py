"""Lane Keeping Assist (LKA) – lateral steering control.

The LKA controller keeps the ego vehicle centred in its current lane by
computing a steering correction from two sources:

1. **Lateral error**: the signed distance (metres) between the vehicle
   centre and the target lane centre.  Positive error means the vehicle
   is to the right of centre; negative means to the left.

2. **Heading error**: the vehicle heading angle theta (rad) relative to
   the road.  A non-zero heading means the vehicle will drift even if it
   is temporarily centred, so the heading error acts as a feed-forward
   damping term.

The combined error is fed to a PID controller whose output is the
steering angle correction in radians, clamped to
±LKA_MAX_STEER_CORRECTION_RAD.
"""

from __future__ import annotations

import math

from src import config
from src.controllers.pid import PIDController


# Weighting factor for the heading error component.
_HEADING_WEIGHT: float = 1.8


class LKA:
    """Lane Keeping Assist controller.

    Computes a steering angle correction to keep the ego vehicle in the
    centre of its current lane.
    """

    def __init__(self) -> None:
        self._pid = PIDController(
            kp=config.LKA_PID_KP,
            ki=config.LKA_PID_KI,
            kd=config.LKA_PID_KD,
            output_min=-config.LKA_MAX_STEER_CORRECTION_RAD,
            output_max=config.LKA_MAX_STEER_CORRECTION_RAD,
            windup_limit=5.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        ego_x: float,
        ego_theta: float,
        lane_center_x: float,
        dt: float,
    ) -> float:
        """Return the steering angle correction in radians.

        A positive correction steers to the right (+x direction), a
        negative correction steers to the left (-x direction).

        The combined error is defined as:

            error = (lane_center_x - ego_x)  [lateral error]
                  - heading_weight * theta    [heading damping]

        Negating theta means that when the vehicle nose points right
        (positive theta), the correction steers left (reduces theta),
        which is the stabilising direction.

        Args:
            ego_x:        Lateral world position of the ego vehicle (m).
            ego_theta:    Heading angle of the ego vehicle (rad).
            lane_center_x: World-x of the target lane centre (m).
            dt:           Time step (s).

        Returns:
            Steering correction in radians, clamped to
            ±LKA_MAX_STEER_CORRECTION_RAD.
        """
        lateral_error = lane_center_x - ego_x

        # Normalise heading to (-pi, pi) before use.
        heading = (ego_theta + math.pi) % (2.0 * math.pi) - math.pi
        combined_error = lateral_error - _HEADING_WEIGHT * heading

        return self._pid.update(combined_error, dt)

    def reset(self) -> None:
        """Reset the internal PID state."""
        self._pid.reset()
