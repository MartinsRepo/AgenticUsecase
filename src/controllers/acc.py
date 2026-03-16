"""Adaptive Cruise Control (ACC) – longitudinal speed regulation.

The ACC controller uses two cascaded PID loops:

1. **Gap control**: if a lead vehicle is detected within the radar range,
   the target speed is scaled down so that the desired time-headway gap is
   maintained.  The gap error drives a P-controller whose output replaces
   the driver-requested target speed.

2. **Speed control**: a PID loop tracks the (possibly modified) target
   speed by producing a longitudinal acceleration command.

The output is an acceleration in m/s² that is handed directly to the
vehicle model.
"""

from __future__ import annotations

from typing import Optional

from src import config
from src.controllers.pid import PIDController


class ACC:
    """Adaptive Cruise Control.

    Args:
        target_speed: Initial driver-requested speed (m/s).
    """

    def __init__(self, target_speed: float = config.ACC_DEFAULT_TARGET_SPEED_MS) -> None:
        self.target_speed = target_speed

        self._speed_pid = PIDController(
            kp=config.ACC_PID_KP,
            ki=config.ACC_PID_KI,
            kd=config.ACC_PID_KD,
            output_min=-config.MAX_DECEL_MS2,
            output_max=config.MAX_ACCEL_MS2,
            windup_limit=20.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        ego_speed: float,
        dt: float,
        lead_distance: Optional[float] = None,
        lead_speed: Optional[float] = None,
    ) -> float:
        """Return longitudinal acceleration command (m/s²).

        When a lead vehicle is detected (*lead_distance* is not None), the
        desired following gap is computed from the current speed multiplied
        by the configured time-headway.  The effective target speed is then
        reduced proportionally to the gap error so that the ego vehicle
        slows down smoothly before reaching the minimum safe distance.

        Args:
            ego_speed:     Current ego speed (m/s).
            dt:            Time step (s).
            lead_distance: Distance to the nearest lead vehicle (m), or None.
            lead_speed:    Speed of the lead vehicle (m/s), or None.

        Returns:
            Longitudinal acceleration command clamped to
            [-MAX_DECEL_MS2, MAX_ACCEL_MS2].
        """
        effective_target = self.target_speed

        if lead_distance is not None:
            # Desired gap grows with speed (time-headway model).
            desired_gap = max(
                config.ACC_SAFE_DISTANCE_M,
                ego_speed * config.ACC_TIME_HEADWAY_S,
            )
            gap_error = lead_distance - desired_gap

            if gap_error < 0.0:
                # We are too close: clamp effective target speed down.
                # Scale by how severely we are violating the desired gap.
                scale = max(0.0, 1.0 + gap_error / desired_gap)
                effective_target = min(effective_target, ego_speed * scale)
                # Hard stop if dangerously close.
                if lead_distance < config.ACC_WARNING_DISTANCE_M:
                    effective_target = 0.0

        speed_error = effective_target - ego_speed
        return self._speed_pid.update(speed_error, dt)

    def reset(self) -> None:
        """Reset the internal PID state."""
        self._speed_pid.reset()

    @property
    def warning_active(self) -> bool:
        """True when the ACC is in hard-braking / warning state."""
        return self._speed_pid._prev_error < -(config.MAX_DECEL_MS2 * 0.5)
