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
    tip_rocker_length: float = 150.0    # mm from tip to rocker/camber junction
    tip_rocker_height: float = 30.0     # mm rise at tip end

    # Camber underfoot
    camber_amount: float = 5.0          # mm rise at centre (positive = arch up)

    # Tail rocker
    tail_rocker_length: float = 150.0   # mm from tail to rocker/camber junction
    tail_rocker_height: float = 20.0    # mm rise at tail end

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
    Compute camber line as three spline sections.

    Section 1 (tip rocker): y=0 → y=tip_rocker_length
      - Free tangent at the tip end, horizontal tangent at the junction (z=0).
    Section 2 (camber): y=tip_rocker_length → y=ski_length-tail_rocker_length
      - Horizontal tangent at both ends (z=0) and at the centre peak.
    Section 3 (tail rocker): y=ski_length-tail_rocker_length → y=ski_length
      - Horizontal tangent at the junction (z=0), free tangent at the tail end.

    Returns:
        (y_points, z_points) where z is vertical height above snow contact.
    """
    tip_junc = params.tip_rocker_length
    tail_junc = ski_length - params.tail_rocker_length
    center_y = (tip_junc + tail_junc) / 2.0

    # Guard against degenerate params
    tip_junc = min(tip_junc, ski_length * 0.4)
    tail_junc = max(tail_junc, ski_length * 0.6)

    # ── Section 1: tip rocker ─────────────────────────────────────────────────
    # y: 0 → tip_junc,  z: tip_height → 0
    # Horizontal tangent (dz/dy = 0) at the junction end.
    # At the tip end we let the spline be natural (not constrained).
    tip_y = np.array([0.0, tip_junc])
    tip_z = np.array([params.tip_rocker_height, 0.0])
    # bc_type: (order, value) — first derivative at each end
    tip_spline = CubicSpline(tip_y, tip_z, bc_type=((2, 0.0), (1, 0.0)))

    # ── Section 2: camber ─────────────────────────────────────────────────────
    # Endpoints z=0, centre peak z=camber_amount, all with horizontal tangents.
    camber_y = np.array([tip_junc, center_y, tail_junc])
    camber_z = np.array([0.0, params.camber_amount, 0.0])
    # Enforce horizontal first derivative at all three points using known-derivative spline
    camber_spline = CubicSpline(
        camber_y, camber_z,
        bc_type=((1, 0.0), (1, 0.0)),
    )

    # ── Section 3: tail rocker ────────────────────────────────────────────────
    tail_y = np.array([tail_junc, ski_length])
    tail_z = np.array([0.0, params.tail_rocker_height])
    tail_spline = CubicSpline(tail_y, tail_z, bc_type=((1, 0.0), (2, 0.0)))

    # ── Sample ────────────────────────────────────────────────────────────────
    tip_pts = np.linspace(0.0, tip_junc, 80)
    camber_pts = np.linspace(tip_junc, tail_junc, 120)
    tail_pts = np.linspace(tail_junc, ski_length, 80)

    y_out = np.concatenate([tip_pts, camber_pts[1:], tail_pts[1:]])
    z_out = np.concatenate([
        tip_spline(tip_pts),
        camber_spline(camber_pts[1:]),
        tail_spline(tail_pts[1:]),
    ])

    return y_out, z_out
