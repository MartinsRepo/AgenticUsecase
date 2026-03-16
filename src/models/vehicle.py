"""Vehicle physics model using the kinematic Bicycle Model.

The bicycle model collapses the two axles into single front/rear contact
points, which is accurate for low-slip highway driving scenarios.

Coordinate convention
---------------------
* World x  – lateral axis (positive = right on screen).
* World y  – longitudinal axis (positive = forward / away from camera).
* theta    – heading angle in radians.  theta = 0 means the vehicle points
              in the +y direction (straight ahead).  Positive theta rotates
              the nose to the right (+x side).

Equations of motion (kinematic bicycle model)
---------------------------------------------
    dx/dt     = v * sin(theta)
    dy/dt     = v * cos(theta)
    dtheta/dt = v * tan(delta) / L
    dv/dt     = a

where delta is the front-wheel steering angle and L is the wheelbase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src import config


@dataclass
class VehicleState:
    """Mutable state vector of a vehicle."""

    x: float = 0.0      # lateral position (m)
    y: float = 0.0      # longitudinal position (m)
    theta: float = 0.0  # heading angle (rad)
    v: float = 0.0      # speed (m/s)

    def to_array(self) -> np.ndarray:
        """Return state as a NumPy array [x, y, theta, v]."""
        return np.array([self.x, self.y, self.theta, self.v], dtype=float)


class Vehicle:
    """Kinematic bicycle model vehicle with clamped inputs.

    Args:
        x:          Initial lateral position in metres.
        y:          Initial longitudinal position in metres.
        theta:      Initial heading angle in radians.
        v:          Initial speed in m/s.
        wheelbase:  Distance between front and rear axle (m).
        width:      Vehicle body width (m).
        length:     Vehicle body length (m).
        max_steer:  Maximum front-wheel steering angle (rad).
        max_speed:  Maximum allowed speed (m/s).
        max_accel:  Maximum longitudinal acceleration (m/s²).
        max_decel:  Maximum longitudinal deceleration (m/s²).
        color:      RGB tuple used for rendering.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        v: float = 0.0,
        wheelbase: float = config.WHEELBASE_M,
        width: float = config.EGO_WIDTH_M,
        length: float = config.EGO_LENGTH_M,
        max_steer: float = config.MAX_STEER_RAD,
        max_speed: float = config.MAX_SPEED_MS,
        max_accel: float = config.MAX_ACCEL_MS2,
        max_decel: float = config.MAX_DECEL_MS2,
        color: tuple[int, int, int] = config.COLOR_EGO,
    ) -> None:
        self.state = VehicleState(x=x, y=y, theta=theta, v=v)
        self.wheelbase = wheelbase
        self.width = width
        self.length = length
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.color = color

        # Last applied control inputs (stored for telemetry/display).
        self.steering_angle: float = 0.0
        self.acceleration: float = 0.0

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def update(self, dt: float, steering_angle: float, acceleration: float) -> None:
        """Advance vehicle state by *dt* seconds using Euler integration.

        Args:
            dt:             Time step in seconds.
            steering_angle: Front-wheel steering angle (rad).  Positive = right.
            acceleration:   Longitudinal acceleration (m/s²).  Positive = forward.
        """
        delta = float(np.clip(steering_angle, -self.max_steer, self.max_steer))
        a = float(np.clip(acceleration, -self.max_decel, self.max_accel))

        self.steering_angle = delta
        self.acceleration = a

        s = self.state
        v = s.v

        # Kinematic bicycle model derivatives.
        dx = v * math.sin(s.theta)
        dy = v * math.cos(s.theta)
        dtheta = (v * math.tan(delta) / self.wheelbase) if abs(v) > 1e-3 else 0.0
        dv = a

        # Euler integration.
        s.x += dx * dt
        s.y += dy * dt
        s.theta += dtheta * dt
        s.v = float(np.clip(s.v + dv * dt, 0.0, self.max_speed))

        # Keep heading in (-pi, pi].
        s.theta = (s.theta + math.pi) % (2.0 * math.pi) - math.pi

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def x(self) -> float:
        """Lateral position (m)."""
        return self.state.x

    @property
    def y(self) -> float:
        """Longitudinal position (m)."""
        return self.state.y

    @property
    def theta(self) -> float:
        """Heading angle (rad)."""
        return self.state.theta

    @property
    def v(self) -> float:
        """Speed (m/s)."""
        return self.state.v

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def get_corners(self) -> list[tuple[float, float]]:
        """Return the four body corners in world (x, y) coordinates.

        The rotation formula for a point (lx, ly) in the vehicle's local
        frame (lateral × longitudinal) into world coordinates is:

            world_x = cx + lx * cos(theta) + ly * sin(theta)
            world_y = cy - lx * sin(theta) + ly * cos(theta)

        where (cx, cy) is the vehicle centre and theta is the heading.
        """
        half_w = self.width / 2.0
        half_l = self.length / 2.0
        cos_t = math.cos(self.state.theta)
        sin_t = math.sin(self.state.theta)

        # Local (lateral, longitudinal) offsets for each corner.
        local_corners: list[tuple[float, float]] = [
            (-half_w,  half_l),   # front-left
            ( half_w,  half_l),   # front-right
            ( half_w, -half_l),   # rear-right
            (-half_w, -half_l),   # rear-left
        ]

        world_corners: list[tuple[float, float]] = []
        for lx, ly in local_corners:
            wx = self.state.x + lx * cos_t + ly * sin_t
            wy = self.state.y - lx * sin_t + ly * cos_t
            world_corners.append((wx, wy))

        return world_corners

    def get_front_center(self) -> tuple[float, float]:
        """Return the world position of the front bumper centre."""
        half_l = self.length / 2.0
        wx = self.state.x + half_l * math.sin(self.state.theta)
        wy = self.state.y + half_l * math.cos(self.state.theta)
        return wx, wy
