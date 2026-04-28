"""Base material design: metal edge layout and cutouts."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np


@dataclass
class BaseParams:
    """Metal edge parameters for ski base."""
    # Edge dimensions
    edge_width: float = 2.0         # mm, width of metal edge
    tip_offset: float = 100.0       # mm from tip where edge starts
    tail_offset: float = 100.0      # mm from tail where edge ends

    # Base material
    base_length: float = 2000.0     # mm
    base_width: float = 100.0       # mm
    base_thickness: float = 2.0     # mm (default for base material)

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


def compute_edge_cutouts(outline: np.ndarray, params: BaseParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute left and right edge cutout profiles.

    Args:
        outline: Nx2 array of (x, y) coordinates in ski space
        params: BaseParams with edge geometry

    Returns:
        (left_edge_outline, right_edge_outline) as Nx2 arrays
    """
    if outline is None or len(outline) == 0:
        return np.array([]), np.array([])

    y_min, y_max = outline[:, 1].min(), outline[:, 1].max()
    ski_length = y_max - y_min

    # Edge active region: from tip_offset to (ski_length - tail_offset)
    edge_start_y = y_min + params.tip_offset
    edge_end_y = y_max - params.tail_offset

    if edge_start_y >= edge_end_y:
        return np.array([]), np.array([])

    # Sample edge region
    n_samples = len(outline)
    y_edge_samples = np.linspace(edge_start_y, edge_end_y, n_samples)

    left_edges = []
    right_edges = []

    for y in y_edge_samples:
        # Find left and right outline points at this y
        x_at_y_candidates = outline[np.abs(outline[:, 1] - y) < 1.0]
        if len(x_at_y_candidates) > 0:
            x_left = np.min(x_at_y_candidates[:, 0])
            x_right = np.max(x_at_y_candidates[:, 0])

            # Edge cutout is inset by edge_width
            left_edges.append([x_left + params.edge_width, y])
            right_edges.append([x_right - params.edge_width, y])

    left_edge_outline = np.array(left_edges) if left_edges else np.array([])
    right_edge_outline = np.array(right_edges) if right_edges else np.array([])

    return left_edge_outline, right_edge_outline


def export_base_dxf(outline: np.ndarray, params: BaseParams, filepath: str | Path) -> None:
    """Export base outline with edge cutouts to DXF file."""
    try:
        import ezdxf
    except ImportError:
        raise ImportError("ezdxf required for DXF export")

    doc = ezdxf.new()
    msp = doc.modelspace()

    # Draw main outline
    if len(outline) > 0:
        points = [(p[0], p[1]) for p in outline]
        msp.add_lwpolyline(points, close=True)

    # Draw edge cutout regions
    left_edge, right_edge = compute_edge_cutouts(outline, params)
    if len(left_edge) > 0:
        points = [(p[0], p[1]) for p in left_edge]
        msp.add_lwpolyline(points)
    if len(right_edge) > 0:
        points = [(p[0], p[1]) for p in right_edge]
        msp.add_lwpolyline(points)

    doc.saveas(filepath)
