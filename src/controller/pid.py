"""
Generic PID controller.

The controller computes a control output u(t) from an error signal e(t):

    u(t) = Kp * e(t)  +  Ki * integral(e) * dt  +  Kd * (de/dt)

Anti-windup is achieved by clamping the integral term to the output limits.
Derivative kick is avoided by differentiating the measurement (process value)
rather than the error, which prevents large spikes when the setpoint changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PIDController:
    """
    Discrete-time PID controller with anti-windup and derivative filtering.

    Parameters
    ----------
    kp          : Proportional gain.
    ki          : Integral gain.
    kd          : Derivative gain.
    output_min  : Lower bound of the controller output.
    output_max  : Upper bound of the controller output.
    tau         : Low-pass filter time constant for the derivative term (s).
                  Set to 0 to disable filtering.
    """

    kp: float
    ki: float
    kd: float
    output_min: float = -float("inf")
    output_max: float = float("inf")
    tau: float = 0.05  # derivative low-pass filter time constant

    # Internal state
    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_measurement: Optional[float] = field(default=None, init=False, repr=False)
    _filtered_derivative: float = field(default=0.0, init=False, repr=False)

    def compute(
        self,
        setpoint: float,
        measurement: float,
        dt: float,
    ) -> float:
        """
        Compute the next control output.

        Parameters
        ----------
        setpoint    : Desired target value.
        measurement : Current measured process value.
        dt          : Time step in seconds (must be > 0).

        Returns
        -------
        Clamped control output.
        """
        if dt <= 0.0:
            return 0.0

        error = setpoint - measurement

        # --- Proportional term ---
        p_term = self.kp * error

        # --- Integral term with anti-windup ---
        self._integral += error * dt
        i_term = self.ki * self._integral
        # Pre-clamp the integral contribution so it does not grow beyond limits
        i_term = max(self.output_min, min(self.output_max, i_term))

        # --- Derivative term (on measurement, not error) with LP filter ---
        if self._prev_measurement is None:
            raw_derivative = 0.0
        else:
            raw_derivative = -(measurement - self._prev_measurement) / dt

        if self.tau > 0.0:
            # First-order low-pass: y[n] = alpha * y[n-1] + (1-alpha) * x[n]
            alpha = self.tau / (self.tau + dt)
            self._filtered_derivative = (
                alpha * self._filtered_derivative + (1.0 - alpha) * raw_derivative
            )
        else:
            self._filtered_derivative = raw_derivative

        d_term = self.kd * self._filtered_derivative
        self._prev_measurement = measurement

        # --- Sum and clamp output ---
        raw_output = p_term + i_term + d_term
        output = max(self.output_min, min(self.output_max, raw_output))

        # Back-calculate anti-windup: if the output is saturated, reverse the
        # excess that the integral term contributed so it does not keep winding.
        if abs(output - raw_output) > 1e-9 and self.ki != 0.0:
            self._integral -= (raw_output - output) / self.ki

        return output

    def reset(self) -> None:
        """Reset all internal state (call when re-initialising the loop)."""
        self._integral = 0.0
        self._prev_measurement = None
        self._filtered_derivative = 0.0
