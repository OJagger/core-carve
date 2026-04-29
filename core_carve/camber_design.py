"""Camber design: vertical ski shape with rocker and camber sections using Bezier cubics."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np


def _bezier_cubic(p0, p1, p2, p3, n: int = 80) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, np.newaxis]
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3


@dataclass
class CamberParams:
    """Camber design parameters — four cubic Bezier segments."""
    # Section endpoints
    tip_rocker_length: float = 150.0    # mm from tip to rocker/camber junction
    tip_rocker_height: float = 30.0     # mm rise at tip end
    camber_amount: float = 5.0          # mm rise at camber peak (centre)
    tail_rocker_length: float = 150.0   # mm from tail to rocker/camber junction
    tail_rocker_height: float = 20.0    # mm rise at tail end

    # Bezier control arm lengths and offsets
    tip_apex_arm: float = 50.0      # y-distance of seg1 P1 from tip end
    tip_apex_arm_dz: float = 0.0    # z-offset of seg1 P1 relative to tip_rocker_height
    tip_junc_arm: float = 40.0      # arm from tip junction toward tip (P2 of seg 1)
    camber_arm: float = 100.0       # arm from each junction toward camber peak (seg2 P1, seg3 P2)
    camber_peak_arm: float = 100.0  # arm from camber peak outward (seg2 P2, seg3 P1)
    tail_junc_arm: float = 40.0     # arm from tail junction toward tail (P1 of seg 4)
    tail_apex_arm: float = 50.0     # y-distance of seg4 P2 from tail end
    tail_apex_arm_dz: float = 0.0   # z-offset of seg4 P2 relative to tail_rocker_height

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


def compute_camber_line(
    ski_length: float, params: CamberParams
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the camber line from four cubic Bezier segments.

    Segment 1 — tip rocker (tip end → tip junction):
      P0=(0, tip_h), P1=(tip_apex_arm, tip_h),
      P2=(tip_junc-tip_junc_arm, 0), P3=(tip_junc, 0)
    Segment 2 — camber rise (tip junction → peak):
      P0=(tip_junc, 0), P1=(tip_junc+camber_arm, 0),
      P2=(center_y-camber_arm, camber_amount), P3=(center_y, camber_amount)
    Segment 3 — camber fall (peak → tail junction):
      P0=(center_y, camber_amount), P1=(center_y+camber_arm, camber_amount),
      P2=(tail_junc-camber_arm, 0), P3=(tail_junc, 0)
    Segment 4 — tail rocker (tail junction → tail end):
      P0=(tail_junc, 0), P1=(tail_junc+tail_junc_arm, 0),
      P2=(L-tail_apex_arm, tail_h), P3=(L, tail_h)

    Returns (y_points, z_points).
    """
    L = ski_length
    tip_junc = float(np.clip(params.tip_rocker_length, 0, L * 0.4))
    tail_junc = float(np.clip(L - params.tail_rocker_length, L * 0.6, L))
    center_y = (tip_junc + tail_junc) / 2.0
    tip_h = params.tip_rocker_height
    tail_h = params.tail_rocker_height
    ca = params.camber_amount
    half_span = max(1.0, center_y - tip_junc)

    tip_apex = min(params.tip_apex_arm, tip_junc * 0.9)
    tip_ja = min(params.tip_junc_arm, tip_junc * 0.9)
    tail_ja = min(params.tail_junc_arm, (L - tail_junc) * 0.9)
    tail_apex = min(params.tail_apex_arm, (L - tail_junc) * 0.9)
    ca_arm = float(np.clip(params.camber_arm, 1.0, half_span * 0.9))
    ca_peak_arm = float(np.clip(params.camber_peak_arm, 1.0, half_span * 0.9))

    seg1 = _bezier_cubic(
        np.array([0.0, tip_h]),
        np.array([tip_apex, tip_h + params.tip_apex_arm_dz]),
        np.array([tip_junc - tip_ja, 0.0]),
        np.array([tip_junc, 0.0]),
        n=60,
    )
    seg2 = _bezier_cubic(
        np.array([tip_junc, 0.0]),
        np.array([tip_junc + ca_arm, 0.0]),
        np.array([center_y - ca_peak_arm, ca]),
        np.array([center_y, ca]),
        n=60,
    )
    seg3 = _bezier_cubic(
        np.array([center_y, ca]),
        np.array([center_y + ca_peak_arm, ca]),
        np.array([tail_junc - ca_arm, 0.0]),
        np.array([tail_junc, 0.0]),
        n=60,
    )
    seg4 = _bezier_cubic(
        np.array([tail_junc, 0.0]),
        np.array([tail_junc + tail_ja, 0.0]),
        np.array([L - tail_apex, tail_h + params.tail_apex_arm_dz]),
        np.array([L, tail_h]),
        n=60,
    )

    pts = np.vstack([seg1, seg2[1:], seg3[1:], seg4[1:]])
    return pts[:, 0], pts[:, 1]


def bezier_control_points(
    ski_length: float, params: CamberParams
) -> dict:
    """
    Return all Bezier control points for the four segments as a dict of named points.

    Keys: seg1_p0, seg1_p1, seg1_p2, seg1_p3,
          seg2_p0, seg2_p1, seg2_p2, seg2_p3,
          seg3_p0, seg3_p1, seg3_p2, seg3_p3,
          seg4_p0, seg4_p1, seg4_p2, seg4_p3
    Each value is (y, z).
    """
    L = ski_length
    tip_junc = float(np.clip(params.tip_rocker_length, 0, L * 0.4))
    tail_junc = float(np.clip(L - params.tail_rocker_length, L * 0.6, L))
    center_y = (tip_junc + tail_junc) / 2.0
    tip_h = params.tip_rocker_height
    tail_h = params.tail_rocker_height
    ca = params.camber_amount
    half_span = max(1.0, center_y - tip_junc)
    tip_apex = min(params.tip_apex_arm, tip_junc * 0.9)
    tip_ja = min(params.tip_junc_arm, tip_junc * 0.9)
    tail_ja = min(params.tail_junc_arm, (L - tail_junc) * 0.9)
    tail_apex = min(params.tail_apex_arm, (L - tail_junc) * 0.9)
    ca_arm = float(np.clip(params.camber_arm, 1.0, half_span * 0.9))
    ca_peak_arm = float(np.clip(params.camber_peak_arm, 1.0, half_span * 0.9))

    return {
        "seg1_p0": (0.0, tip_h),
        "seg1_p1": (tip_apex, tip_h + params.tip_apex_arm_dz),
        "seg1_p2": (tip_junc - tip_ja, 0.0),
        "seg1_p3": (tip_junc, 0.0),
        "seg2_p0": (tip_junc, 0.0),
        "seg2_p1": (tip_junc + ca_arm, 0.0),
        "seg2_p2": (center_y - ca_peak_arm, ca),
        "seg2_p3": (center_y, ca),
        "seg3_p0": (center_y, ca),
        "seg3_p1": (center_y + ca_peak_arm, ca),
        "seg3_p2": (tail_junc - ca_arm, 0.0),
        "seg3_p3": (tail_junc, 0.0),
        "seg4_p0": (tail_junc, 0.0),
        "seg4_p1": (tail_junc + tail_ja, 0.0),
        "seg4_p2": (L - tail_apex, tail_h + params.tail_apex_arm_dz),
        "seg4_p3": (L, tail_h),
    }
