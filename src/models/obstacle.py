"""Obstacle model – both static road obstacles and moving NPC vehicles."""

from __future__ import annotations

import random
from typing import Tuple

from src import config


class Obstacle:
    """A static obstacle placed by the user on the road.

    Attributes:
        x:      Lateral world position (m).
        y:      Longitudinal world position (m).
        width:  Body width (m).
        length: Body length (m).
        color:  RGB rendering colour.
    """

    def __init__(
        self,
        x: float,
        y: float,
        width: float = 2.0,
        length: float = 4.5,
        color: Tuple[int, int, int] = config.COLOR_OBSTACLE,
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.length = length
        self.color = color

    def get_corners(self) -> list[Tuple[float, float]]:
        """Return four world-space corners (no heading – always aligned to axes)."""
        half_w = self.width / 2.0
        half_l = self.length / 2.0
        return [
            (self.x - half_w, self.y + half_l),  # front-left
            (self.x + half_w, self.y + half_l),  # front-right
            (self.x + half_w, self.y - half_l),  # rear-right
            (self.x - half_w, self.y - half_l),  # rear-left
        ]


class NPCVehicle:
    """A simple non-player vehicle that drives straight at a fixed speed.

    NPC vehicles do not use ADAS logic – they maintain constant velocity
    in the +y direction (straight ahead) within their assigned lane.

    Args:
        lane_index: Integer lane index (0 = leftmost, NUM_LANES-1 = rightmost).
        y:          Initial longitudinal position (m).
        speed:      Constant forward speed (m/s).
        color:      RGB rendering colour.
    """

    def __init__(
        self,
        lane_index: int,
        y: float,
        speed: float | None = None,
        color: Tuple[int, int, int] = config.COLOR_NPC,
    ) -> None:
        self.lane_index = lane_index
        self.x: float = _lane_center_x(lane_index)
        self.y: float = y
        self.speed: float = (
            speed
            if speed is not None
            else config.NPC_BASE_SPEED_MS + random.uniform(
                -config.NPC_SPEED_SPREAD_MS, config.NPC_SPEED_SPREAD_MS
            )
        )
        self.width: float = config.NPC_WIDTH_M
        self.length: float = config.NPC_LENGTH_M
        self.color: Tuple[int, int, int] = color
        self.theta: float = 0.0  # always heading straight

    def update(self, dt: float) -> None:
        """Advance NPC position by one time step."""
        self.y += self.speed * dt

    def get_corners(self) -> list[Tuple[float, float]]:
        """Return four world-space corners (axis-aligned)."""
        half_w = self.width / 2.0
        half_l = self.length / 2.0
        return [
            (self.x - half_w, self.y + half_l),
            (self.x + half_w, self.y + half_l),
            (self.x + half_w, self.y - half_l),
            (self.x - half_w, self.y - half_l),
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lane_center_x(lane_index: int) -> float:
    """Return the world-x centre of the given lane index.

    Lane 0 is the leftmost lane; lane (NUM_LANES - 1) is the rightmost.
    The road is centred at world x = 0.
    """
    total_width = config.NUM_LANES * config.LANE_WIDTH_M
    left_edge = -total_width / 2.0
    return left_edge + (lane_index + 0.5) * config.LANE_WIDTH_M
