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

    The camber line is the bottom surface of the ski. It touches snow at the
    tip and tail rocker ends (zero gradient), and arches up in the middle (camber).

    Args:
        ski_length: Total ski length (mm)
        params: CamberParams

    Returns:
        (y_points, z_points) where y is along ski, z is vertical (up from snow contact)
    """
    # Key points along ski length (where rocker and camber sections meet)
    tip_rocker_end_y = params.tip_rocker_length
    tail_rocker_start_y = ski_length - params.tail_rocker_length
    center_y = ski_length / 2.0

    # Vertical positions: z=0 at rocker/camber boundaries (where ski touches snow)
    # Rocker endpoints have z=0 (snow contact with zero gradient)
    rocker_contact_z = 0.0

    # But the rockers themselves arc up from the ends (tip and tail extremes)
    tip_extent_z = params.tip_rocker_height
    tail_extent_z = params.tail_rocker_height
    center_z = params.camber_amount

    # Create control points for 3 splines
    # Tip rocker: from tip (raised) to rocker start (touches snow, z=0)
    # Curve down from tip_extent toward contact at rocker_end
    tip_y = np.array([0.0, params.tip_rocker_length / 3.0, params.tip_rocker_length * 2 / 3.0, tip_rocker_end_y])
    tip_z = np.array([tip_extent_z, params.tip_rocker_height * 0.7, params.tip_rocker_height * 0.3, rocker_contact_z])

    # Camber section: from rocker start (z=0) through center peak, to rocker end (z=0)
    camber_y = np.array([tip_rocker_end_y, center_y, tail_rocker_start_y])
    camber_z = np.array([rocker_contact_z, center_z, rocker_contact_z])

    # Tail rocker: from rocker start (touches snow, z=0) to tail (raised)
    # Curve up from contact to tail_extent
    tail_y = np.array([tail_rocker_start_y, tail_rocker_start_y + (ski_length - tail_rocker_start_y) / 3.0,
                       tail_rocker_start_y + 2 * (ski_length - tail_rocker_start_y) / 3.0, ski_length])
    tail_z = np.array([rocker_contact_z, tail_extent_z * 0.3, tail_extent_z * 0.7, tail_extent_z])

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
