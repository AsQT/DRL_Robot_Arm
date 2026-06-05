"""
Waypoint trajectory utilities.

Helpers for building, serialising, and deserialising waypoint trajectories
produced by the DRL policy or exported for ROS2 / MoveIt consumption.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
class Waypoint:
    """A single waypoint in a trajectory."""

    __slots__ = (
        "step",
        "x",
        "y",
        "z",
        "distance_to_target",
        "action_x",
        "action_y",
        "action_z",
        "reward",
        "is_success",
        "is_collision",
        "is_out_of_workspace",
    )

    def __init__(
        self,
        step: int,
        x: float,
        y: float,
        z: float,
        distance_to_target: float,
        action_x: float = 0.0,
        action_y: float = 0.0,
        action_z: float = 0.0,
        reward: float = 0.0,
        is_success: bool = False,
        is_collision: bool = False,
        is_out_of_workspace: bool = False,
    ) -> None:
        self.step = step
        self.x = x
        self.y = y
        self.z = z
        self.distance_to_target = distance_to_target
        self.action_x = action_x
        self.action_y = action_y
        self.action_z = action_z
        self.reward = reward
        self.is_success = is_success
        self.is_collision = is_collision
        self.is_out_of_workspace = is_out_of_workspace

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "distance_to_target": float(self.distance_to_target),
            "action_x": float(self.action_x),
            "action_y": float(self.action_y),
            "action_z": float(self.action_z),
            "reward": float(self.reward),
            "is_success": bool(self.is_success),
            "is_collision": bool(self.is_collision),
            "is_out_of_workspace": bool(self.is_out_of_workspace),
        }

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)


class Trajectory:
    """A list of Waypoints forming a complete episode trajectory."""

    def __init__(self) -> None:
        self.waypoints: List[Waypoint] = []

    def add(
        self,
        step: int,
        position: np.ndarray,
        distance_to_target: float,
        action: np.ndarray | None = None,
        reward: float = 0.0,
        is_success: bool = False,
        is_collision: bool = False,
        is_out_of_workspace: bool = False,
    ) -> None:
        action = action if action is not None else np.zeros(3)
        self.waypoints.append(
            Waypoint(
                step=step,
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
                distance_to_target=distance_to_target,
                action_x=float(action[0]),
                action_y=float(action[1]),
                action_z=float(action[2]),
                reward=reward,
                is_success=is_success,
                is_collision=is_collision,
                is_out_of_workspace=is_out_of_workspace,
            )
        )

    def to_csv(self, path: Path) -> None:
        """Serialise the trajectory as a CSV file."""
        fieldnames = [
            "step", "x", "y", "z", "distance_to_target",
            "action_x", "action_y", "action_z",
            "reward", "is_success", "is_collision", "is_out_of_workspace",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for wp in self.waypoints:
                writer.writerow(wp.to_dict())

    def to_json(self, path: Path, metadata: Dict[str, Any] | None = None) -> None:
        """Serialise the trajectory as a JSON file with optional metadata."""
        data: Dict[str, Any] = {
            "waypoints": [wp.to_dict() for wp in self.waypoints],
        }
        if metadata is not None:
            data["metadata"] = metadata
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)


# --------------------------------------------------------------------------- #
# Load helpers
# --------------------------------------------------------------------------- #
def load_waypoints_from_csv(path: Path) -> Trajectory:
    """Load a trajectory from a CSV file."""
    traj = Trajectory()
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            traj.add(
                step=int(row["step"]),
                position=np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                distance_to_target=float(row["distance_to_target"]),
                action=np.array([float(row["action_x"]), float(row["action_y"]), float(row["action_z"])]),
                reward=float(row["reward"]),
                is_success=row["is_success"] == "True",
                is_collision=row["is_collision"] == "True",
                is_out_of_workspace=row["is_out_of_workspace"] == "True",
            )
    return traj
