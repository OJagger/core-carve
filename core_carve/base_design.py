"""Base material design: metal edge layout, step cutouts, and G-code."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class BaseParams:
    """Parameters for ski base design including metal edge cutouts."""
    # Edge dimensions
    edge_width: float = 2.0         # mm, step depth (inset) for metal edge groove
    tip_offset: float = 100.0       # mm from tip where edge starts
    tail_offset: float = 100.0      # mm from tail where edge ends

    # Base material
    base_length: float = 2000.0     # mm
    base_width: float = 100.0       # mm
    base_thickness: float = 2.0     # mm

    # Cutting tool
    cutter_type: str = "router"     # "router" or "drag_knife"
    tool_diameter: float = 6.0      # mm (router)
    spindle_speed: float = 18000.0  # RPM (router)
    cutting_feed: float = 2000.0    # mm/min
    plunge_feed: float = 300.0      # mm/min (router)
    clearance_height: float = 10.0  # mm above material (router)
    kerf_width: float = 0.5         # mm (drag knife)
    cutting_speed: float = 1000.0   # mm/min (drag knife)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "BaseParams":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _sample_outline_half_widths(outline: np.ndarray, y_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each y in y_samples, interpolate the left (min x) and right (max x) outline x-coordinates.

    The outline is a closed polygon. We split it into left and right halves by
    finding the extreme x points and sorting each half by y.
    """
    y = outline[:, 1]
    x = outline[:, 0]

    # Build separate left/right half by sorting points
    # Find the tip (min y) and tail (max y) indices
    tip_idx = np.argmin(y)
    tail_idx = np.argmax(y)

    # Walk around the polygon in both directions from tip to tail
    n = len(outline)

    # Direction 1: tip_idx → ... → tail_idx (going "right" around polygon)
    half1_indices = []
    i = tip_idx
    while True:
        half1_indices.append(i)
        if i == tail_idx:
            break
        i = (i + 1) % n

    # Direction 2: tip_idx → ... → tail_idx (going "left" around polygon)
    half2_indices = []
    i = tip_idx
    while True:
        half2_indices.append(i)
        if i == tail_idx:
            break
        i = (i - 1) % n

    half1 = outline[half1_indices]
    half2 = outline[half2_indices]

    # Determine which half is "left" (lower x at mid-y) and which is "right"
    mid_y = (y.min() + y.max()) / 2
    def x_at_mid(half):
        dists = np.abs(half[:, 1] - mid_y)
        return half[np.argmin(dists), 0]

    if x_at_mid(half1) < x_at_mid(half2):
        left_half, right_half = half1, half2
    else:
        left_half, right_half = half2, half1

    # Sort each half by y for interpolation
    left_sorted = left_half[np.argsort(left_half[:, 1])]
    right_sorted = right_half[np.argsort(right_half[:, 1])]

    left_x = np.interp(y_samples, left_sorted[:, 1], left_sorted[:, 0])
    right_x = np.interp(y_samples, right_sorted[:, 1], right_sorted[:, 0])

    return left_x, right_x


def compute_base_outline(outline: np.ndarray, params: BaseParams) -> np.ndarray:
    """
    Compute the closed base outline polygon with metal edge step cutouts.

    The base outline follows the ski outline, but in the edge-active region
    (between tip_offset and tail_offset from the ends) the edge of the base
    is inset by `edge_width` to create a groove for the metal edge.
    Steps (vertical transitions) connect the inset and full-width sections.

    Returns:
        Nx2 array of (x, y) coords forming a closed polygon (tip→right side→tail→left side)
    """
    if outline is None or len(outline) == 0:
        return np.array([]).reshape(0, 2)

    y_min = outline[:, 1].min()
    y_max = outline[:, 1].max()

    edge_start_y = y_min + params.tip_offset
    edge_end_y = y_max - params.tail_offset

    # Sample the full outline at fine resolution
    n = 500
    y_full = np.linspace(y_min, y_max, n)
    left_full, right_full = _sample_outline_half_widths(outline, y_full)

    # Sample the edge-active region (inset)
    n_edge = 400
    y_edge = np.linspace(edge_start_y, edge_end_y, n_edge)
    left_edge, right_edge = _sample_outline_half_widths(outline, y_edge)
    left_inset = left_edge + params.edge_width
    right_inset = right_edge - params.edge_width

    # ── Build right side (tip → tail) ────────────────────────────────────────
    # Tip section: full width, from y_min to edge_start_y
    tip_mask = y_full <= edge_start_y
    right_tip_y = y_full[tip_mask]
    right_tip_x = right_full[tip_mask]

    # Step at edge_start_y: from full-width to inset
    right_at_edge_start_full = np.interp(edge_start_y, y_full, right_full)
    right_at_edge_start_inset = np.interp(edge_start_y, y_edge, right_inset)

    # Edge section: inset right side
    right_edge_pts = np.column_stack([right_inset, y_edge])

    # Step at edge_end_y: from inset back to full-width
    right_at_edge_end_inset = np.interp(edge_end_y, y_edge, right_inset)
    right_at_edge_end_full = np.interp(edge_end_y, y_full, right_full)

    # Tail section: full width, from edge_end_y to y_max
    tail_mask = y_full >= edge_end_y
    right_tail_y = y_full[tail_mask]
    right_tail_x = right_full[tail_mask]

    # Assemble right side
    right_side = np.vstack([
        np.column_stack([right_tip_x, right_tip_y]),
        [[right_at_edge_start_full, edge_start_y]],
        [[right_at_edge_start_inset, edge_start_y]],
        right_edge_pts,
        [[right_at_edge_end_inset, edge_end_y]],
        [[right_at_edge_end_full, edge_end_y]],
        np.column_stack([right_tail_x, right_tail_y]),
    ])

    # ── Build left side (tail → tip, reversed) ───────────────────────────────
    left_at_edge_start_full = np.interp(edge_start_y, y_full, left_full)
    left_at_edge_start_inset = np.interp(edge_start_y, y_edge, left_inset)
    left_at_edge_end_inset = np.interp(edge_end_y, y_edge, left_inset)
    left_at_edge_end_full = np.interp(edge_end_y, y_full, left_full)

    left_side = np.vstack([
        np.column_stack([left_full[tail_mask], y_full[tail_mask]])[::-1],  # tail: y_max→edge_end_y
        [[left_at_edge_end_full, edge_end_y]],
        [[left_at_edge_end_inset, edge_end_y]],
        np.column_stack([left_inset, y_edge])[::-1],              # edge section inset, reversed
        [[left_at_edge_start_inset, edge_start_y]],
        [[left_at_edge_start_full, edge_start_y]],
        np.column_stack([left_full[tip_mask], y_full[tip_mask]])[::-1],    # tip: edge_start_y→y_min
    ])

    # Combine: right side goes tip→tail, left side goes tail→tip
    polygon = np.vstack([right_side, left_side])
    return polygon


def export_base_dxf(outline: np.ndarray, params: BaseParams, filepath: str | Path) -> None:
    """Export base outline with edge step cutouts to DXF file."""
    try:
        import ezdxf
    except ImportError:
        raise ImportError("ezdxf required for DXF export")

    doc = ezdxf.new()
    msp = doc.modelspace()

    base_polygon = compute_base_outline(outline, params)
    if len(base_polygon) > 0:
        points = [(float(p[0]), float(p[1])) for p in base_polygon]
        msp.add_lwpolyline(points, close=True)

    doc.saveas(filepath)


def compute_base_gcode(outline: np.ndarray, params: BaseParams) -> str:
    """
    Generate G-code to cut the base outline from sheet material.

    For a router: contour cut around the base outline with depth passes.
    For a drag knife: single-pass cut following the outline.
    """
    base_polygon = compute_base_outline(outline, params)
    if len(base_polygon) == 0:
        return ""

    lines = ["G21 G17 G90 G94"]  # metric, XY plane, absolute, feed in mm/min

    if params.cutter_type == "router":
        lines += _router_gcode(base_polygon, params)
    else:
        lines += _drag_knife_gcode(base_polygon, params)

    lines.append("M30")
    return "\n".join(lines)


def _router_gcode(polygon: np.ndarray, params: BaseParams) -> list[str]:
    lines = [
        f"S{params.spindle_speed:.0f} M03",
        f"G00 Z{params.clearance_height:.3f}",
    ]
    # Single finishing pass at full depth (base material is thin plastic)
    depth = -params.base_thickness
    # Rapid to start
    lines.append(f"G00 X{polygon[0, 0]:.3f} Y{polygon[0, 1]:.3f}")
    lines.append(f"G01 Z{depth:.3f} F{params.plunge_feed:.1f}")
    for pt in polygon[1:]:
        lines.append(f"G01 X{pt[0]:.3f} Y{pt[1]:.3f} F{params.cutting_feed:.1f}")
    # Close
    lines.append(f"G01 X{polygon[0, 0]:.3f} Y{polygon[0, 1]:.3f} F{params.cutting_feed:.1f}")
    lines.append(f"G00 Z{params.clearance_height:.3f}")
    lines.append("M05")
    return lines


def _drag_knife_gcode(polygon: np.ndarray, params: BaseParams) -> list[str]:
    lines = [f"G00 Z{params.clearance_height:.3f}"]
    lines.append(f"G00 X{polygon[0, 0]:.3f} Y{polygon[0, 1]:.3f}")
    lines.append("G01 Z0.000 F100.0")  # Lower knife
    for pt in polygon[1:]:
        lines.append(f"G01 X{pt[0]:.3f} Y{pt[1]:.3f} F{params.cutting_speed:.1f}")
    lines.append(f"G01 X{polygon[0, 0]:.3f} Y{polygon[0, 1]:.3f} F{params.cutting_speed:.1f}")
    lines.append(f"G00 Z{params.clearance_height:.3f}")
    return lines
