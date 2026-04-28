"""Camber design: vertical ski shape with rocker and camber sections."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class CamberParams:
    """Camber design parameters for vertical ski shape."""
    # Tip rocker
    tip_rocker_length: float = 150.0    # mm from tip
    tip_rocker_height: float = 30.0     # mm rise from contact point

    # Camber underfoot
    camber_amount: float = 20.0         # mm rise at center (positive = arch)

    # Tail rocker
    tail_rocker_length: float = 150.0   # mm from tail
    tail_rocker_height: float = 30.0    # mm rise from contact point

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "CamberParams":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def compute_camber_line(ski_length: float, params: CamberParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute camber line with 3 spline sections: tip rocker, camber, tail rocker.

    Args:
        ski_length: Total ski length (mm)
        params: CamberParams

    Returns:
        (y_points, z_points) where y is along ski, z is vertical
    """
    # Key points along ski length
    tip_end_y = params.tip_rocker_length
    tail_start_y = ski_length - params.tail_rocker_length
    center_y = ski_length / 2.0

    # Vertical positions (Z axis)
    # Contact points at tip and tail ends (z=0)
    tip_contact_z = 0.0
    tail_contact_z = 0.0

    # Intermediate points
    tip_end_z = params.tip_rocker_height
    tail_start_z = params.tail_rocker_height
    center_z = params.camber_amount

    # Create control points for 3 splines
    # Tip rocker: from (0, tip_contact) to (tip_end_y, tip_end_z)
    tip_y = np.array([0.0, params.tip_rocker_length / 3.0, params.tip_rocker_length * 2 / 3.0, tip_end_y])
    tip_z = np.array([tip_contact_z, params.tip_rocker_height / 2.0, params.tip_rocker_height * 0.8, tip_end_z])

    # Camber section: from (tip_end_y, tip_end_z) to (tail_start_y, tail_start_z)
    camber_y = np.array([tip_end_y, center_y, tail_start_y])
    camber_z = np.array([tip_end_z, center_z, tail_start_z])

    # Tail rocker: from (tail_start_y, tail_start_z) to (ski_length, tail_contact)
    tail_y = np.array([tail_start_y, tail_start_y + (ski_length - tail_start_y) / 3.0,
                       tail_start_y + 2 * (ski_length - tail_start_y) / 3.0, ski_length])
    tail_z = np.array([tail_start_z, params.tail_rocker_height * 0.8, params.tail_rocker_height / 2.0, tail_contact_z])

    # Create splines for each section
    try:
        tip_spline = CubicSpline(tip_y, tip_z, bc_type="natural")
        camber_spline = CubicSpline(camber_y, camber_z, bc_type="natural")
        tail_spline = CubicSpline(tail_y, tail_z, bc_type="natural")

        # Sample splines
        tip_samples = np.linspace(0.0, tip_end_y, 50)
        camber_samples = np.linspace(tip_end_y, tail_start_y, 100)
        tail_samples = np.linspace(tail_start_y, ski_length, 50)

        tip_z_samples = tip_spline(tip_samples)
        camber_z_samples = camber_spline(camber_samples)
        tail_z_samples = tail_spline(tail_samples)

        y_points = np.concatenate([tip_samples, camber_samples[1:], tail_samples[1:]])
        z_points = np.concatenate([tip_z_samples, camber_z_samples[1:], tail_z_samples[1:]])

        return y_points, z_points
    except Exception:
        # Fallback to linear interpolation if spline fails
        all_y = np.concatenate([tip_y, camber_y[1:], tail_y[1:]])
        all_z = np.concatenate([tip_z, camber_z[1:], tail_z[1:]])
        return all_y, all_z
