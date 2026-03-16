"""
Vehicle model using the kinematic bicycle model.

The bicycle model approximates a four-wheeled vehicle by collapsing each
axle into a single virtual wheel at the axle midpoint.  It is accurate at
low-to-moderate speeds and is the standard choice for lane-keeping and
adaptive-cruise simulations.

State vector:
    x        : World-X position of the rear axle midpoint (pixels)
    y        : World-Y position of the rear axle midpoint (pixels)
    heading  : Yaw angle in radians (0 = pointing right, positive CCW)
    speed    : Forward speed in pixels per second

Inputs:
    steering_angle : Front-wheel steering angle in radians
    acceleration   : Longitudinal acceleration in pixels/s^2 (negative = brake)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

WHEELBASE: float = 40.0          # Distance between axles in pixels
MAX_STEERING: float = math.radians(30.0)   # Maximum front-wheel steer angle
MAX_SPEED: float = 300.0          # pixels / second
MIN_SPEED: float = 0.0
MAX_ACCEL: float = 80.0           # pixels / s^2
MAX_DECEL: float = -120.0         # pixels / s^2 (emergency braking)


@dataclass
class VehicleState:
    """Mutable snapshot of a vehicle's kinematic state."""

    x: float = 0.0
    y: float = 0.0
    heading: float = -math.pi / 2.0   # pointing upward in screen coords
    speed: float = 0.0

    def position(self) -> Tuple[float, float]:
        """Return the (x, y) position as a tuple."""
        return (self.x, self.y)

    def front_axle(self, wheelbase: float = WHEELBASE) -> Tuple[float, float]:
        """Compute the world position of the front axle centre."""
        fx = self.x + wheelbase * math.cos(self.heading)
        fy = self.y + wheelbase * math.sin(self.heading)
        return (fx, fy)


@dataclass
class Vehicle:
    """
    Kinematic bicycle model vehicle.

    Parameters
    ----------
    x, y        : Initial world position of the vehicle centre (pixels).
    heading     : Initial heading in radians.
    speed       : Initial forward speed in pixels/second.
    wheelbase   : Distance between front and rear axles in pixels.
    length      : Visual length of the car (pixels).
    width       : Visual width of the car (pixels).
    is_ego      : Whether this is the player-controlled ego vehicle.
    """

    x: float = 400.0
    y: float = 300.0
    heading: float = -math.pi / 2.0
    speed: float = 100.0
    wheelbase: float = WHEELBASE
    length: float = 48.0
    width: float = 24.0
    is_ego: bool = False

    # Internal state
    state: VehicleState = field(init=False)

    def __post_init__(self) -> None:
        self.state = VehicleState(
            x=self.x,
            y=self.y,
            heading=self.heading,
            speed=self.speed,
        )

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        acceleration: float,
        steering_angle: float,
    ) -> None:
        """
        Advance the vehicle state by one time step using the kinematic
        bicycle model update equations.

        Parameters
        ----------
        dt              : Time step in seconds.
        acceleration    : Desired longitudinal acceleration (pixels/s^2).
        steering_angle  : Desired front-wheel steer angle (radians).
        """
        # Clamp inputs to physical limits
        accel = float(np.clip(acceleration, MAX_DECEL, MAX_ACCEL))
        delta = float(np.clip(steering_angle, -MAX_STEERING, MAX_STEERING))

        s = self.state

        # Update speed
        s.speed = float(np.clip(s.speed + accel * dt, MIN_SPEED, MAX_SPEED))

        # Bicycle model kinematics
        # angular_velocity = v * tan(delta) / L
        angular_velocity = (
            s.speed * math.tan(delta) / self.wheelbase if self.wheelbase > 0 else 0.0
        )

        # Integrate position and heading
        s.x += s.speed * math.cos(s.heading) * dt
        s.y += s.speed * math.sin(s.heading) * dt
        s.heading += angular_velocity * dt

        # Normalise heading to [-pi, pi]
        s.heading = math.atan2(math.sin(s.heading), math.cos(s.heading))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def position(self) -> Tuple[float, float]:
        """Current (x, y) position."""
        return self.state.position()

    @property
    def heading_angle(self) -> float:
        """Current heading in radians."""
        return self.state.heading

    @property
    def speed_value(self) -> float:
        """Current forward speed in pixels/second."""
        return self.state.speed

    def speed_kmh(self) -> float:
        """Return speed in km/h (using 1 pixel = 0.1 m as scale)."""
        return self.state.speed * 0.1 * 3.6

    def corners(self) -> list[Tuple[float, float]]:
        """
        Compute the four corner positions of the vehicle bounding box in
        world coordinates, used for rendering and collision detection.
        """
        cx, cy = self.state.x, self.state.y
        h = self.state.heading
        half_l = self.length / 2.0
        half_w = self.width / 2.0

        cos_h = math.cos(h)
        sin_h = math.sin(h)

        # Local offsets for the four corners
        offsets = [
            (+half_l, -half_w),  # front-left
            (+half_l, +half_w),  # front-right
            (-half_l, +half_w),  # rear-right
            (-half_l, -half_w),  # rear-left
        ]
        return [
            (cx + ox * cos_h - oy * sin_h, cy + ox * sin_h + oy * cos_h)
            for ox, oy in offsets
        ]
