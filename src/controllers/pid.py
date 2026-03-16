"""Generic discrete-time PID controller.

The controller operates in the time domain and is discretised using the
backward Euler method.  An integral anti-windup clamp prevents integrator
saturation during large sustained errors.
"""

from __future__ import annotations


class PIDController:
    """Proportional-Integral-Derivative controller.

    Args:
        kp:           Proportional gain.
        ki:           Integral gain.
        kd:           Derivative gain.
        output_min:   Lower clamp on the controller output.
        output_max:   Upper clamp on the controller output.
        windup_limit: Symmetric anti-windup clamp on the integral term.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float = -float("inf"),
        output_max: float = float("inf"),
        windup_limit: float = 50.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.windup_limit = windup_limit

        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._first: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, error: float, dt: float) -> float:
        """Compute one controller output given the current *error*.

        Args:
            error: Setpoint minus measured value.
            dt:    Time step in seconds (must be > 0).

        Returns:
            Clamped controller output.
        """
        if dt <= 0.0:
            return 0.0

        # Derivative term (avoid derivative kick on first call).
        if self._first:
            derivative = 0.0
            self._first = False
        else:
            derivative = (error - self._prev_error) / dt

        # Integral with anti-windup clamping.
        self._integral += error * dt
        self._integral = max(
            -self.windup_limit, min(self.windup_limit, self._integral)
        )

        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(self.output_min, min(self.output_max, output))

    def reset(self) -> None:
        """Reset accumulated state (integral and previous error)."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True
