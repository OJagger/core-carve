"""Ski geometry data model and DXF/JSON I/O."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import ezdxf
from scipy.interpolate import CubicSpline


@dataclass
class SkiParams:
    tip_infill: float = 0.0        # mm — distance from ski tip to core start
    tail_infill: float = 0.0       # mm — distance from ski tail to core end
    sidewall_width: float = 6.0    # mm
    sidewall_overlap: float = 0.0  # mm — how far sidewall protrudes beyond base outline
    tip_thickness: float = 2.0     # mm — core thickness at tip end of core
    underfoot_thickness: float = 10.0  # mm
    tail_thickness: float = 2.0    # mm — core thickness at tail end of core
    underfoot_length: float = 300.0  # mm — flat-thickness region centred on waist

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "SkiParams":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SkiGeometry:
    """Derived geometry computed from planform DXF + SkiParams."""
    # Planform outline as closed polygon (Nx2, mm, tip at origin)
    outline: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    ski_length: float = 0.0          # mm, along Y axis
    waist_x: float = 0.0             # longitudinal position of waist (min width)
    geometric_centre_x: float = 0.0  # longitudinal midpoint
    setback: float = 0.0             # geometric_centre_x - waist_x (positive = waist set back)

    # Derived longitudinal positions (measured from tip = 0)
    core_tip_x: float = 0.0
    core_tail_x: float = 0.0
    underfoot_start_x: float = 0.0
    underfoot_end_x: float = 0.0

    # Thickness profile splines (one for each taper region)
    _thickness_spline_tip: Optional[CubicSpline] = field(default=None, repr=False)
    _thickness_spline_tail: Optional[CubicSpline] = field(default=None, repr=False)
    _underfoot_thickness: float = field(default=0.0, repr=False)

    def width_at_y(self, y: float) -> float:
        """Return the full ski width (left edge to right edge) at a given Y position."""
        xs = _outline_x_at_y(self.outline, y)
        if len(xs) >= 2:
            return float(xs.max() - xs.min())
        return 0.0

    def thickness_at(self, x_arr: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x_arr, dtype=float)

        # Evaluate appropriate spline for each region
        mask_tip = x_arr < self.underfoot_start_x
        if mask_tip.any() and self._thickness_spline_tip is not None:
            result[mask_tip] = self._thickness_spline_tip(x_arr[mask_tip])

        mask_underfoot = (x_arr >= self.underfoot_start_x) & (x_arr <= self.underfoot_end_x)
        result[mask_underfoot] = self._underfoot_thickness

        mask_tail = x_arr > self.underfoot_end_x
        if mask_tail.any() and self._thickness_spline_tail is not None:
            result[mask_tail] = self._thickness_spline_tail(x_arr[mask_tail])

        return result


def load_planform_dxf(path: str | Path) -> np.ndarray:
    """
    Read a DXF file and extract the ski planform outline as an Nx2 array.

    Expects a single closed LWPOLYLINE or a collection of LINE/ARC entities forming
    a closed outline.  The ski tip must be at (0, 0).  The ski is expected to run
    along the Y axis (vertical in the DXF).  Points are reordered so the ski runs
    from Y=0 (tip) upward.
    """
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    pts = _extract_outline_points(msp)
    if pts is None or len(pts) < 3:
        raise ValueError("Could not extract a closed outline from the DXF file.")

    pts = _orient_tip_at_origin(pts)
    return pts


def _extract_outline_points(msp) -> Optional[np.ndarray]:
    # Try LWPOLYLINE first
    for entity in msp:
        if entity.dxftype() == "LWPOLYLINE":
            pts = np.array(list(entity.vertices()))[:, :2]
            if entity.closed:
                pts = np.vstack([pts, pts[0]])
            return pts.astype(float)

    # Try POLYLINE
    for entity in msp:
        if entity.dxftype() == "POLYLINE":
            pts = np.array([(v.dxf.x, v.dxf.y) for v in entity.points])
            if entity.dxf.flags & 0x1:  # closed flag
                pts = np.vstack([pts, pts[0]])
            return pts.astype(float)

    # Fall back: chain LINE, SPLINE, ARC segments into a polyline
    segments = []
    for entity in msp:
        if entity.dxftype() == "LINE":
            segments.append((
                (float(entity.dxf.start.x), float(entity.dxf.start.y)),
                (float(entity.dxf.end.x), float(entity.dxf.end.y)),
            ))
        elif entity.dxftype() == "ARC":
            pts_arc = _sample_arc(entity, n_points=50)
            for i in range(len(pts_arc) - 1):
                segments.append((tuple(pts_arc[i]), tuple(pts_arc[i + 1])))
        elif entity.dxftype() == "SPLINE":
            pts_spline = _sample_spline(entity, n_points=50)
            for i in range(len(pts_spline) - 1):
                segments.append((tuple(pts_spline[i]), tuple(pts_spline[i + 1])))

    if segments:
        pts = _chain_segments(segments)
        if pts is not None:
            pts = np.array(pts, dtype=float)
            # Ensure the polyline is closed
            if len(pts) > 1 and not np.allclose(pts[0], pts[-1], atol=1e-6):
                pts = np.vstack([pts, pts[0]])
            return pts

    return None


def _sample_arc(arc_entity, n_points: int = 100) -> np.ndarray:
    """Sample an ARC entity to produce a polyline of points."""
    try:
        center = np.array([arc_entity.dxf.center.x, arc_entity.dxf.center.y])
        radius = float(arc_entity.dxf.radius)
        start_angle = float(arc_entity.dxf.start_angle)
        end_angle = float(arc_entity.dxf.end_angle)

        # Convert to radians
        start_rad = start_angle * np.pi / 180.0
        end_rad = end_angle * np.pi / 180.0

        # Handle angle wrap: if end < start, the arc goes through 0°/360°
        if end_angle < start_angle:
            # Arc crosses 0°, so sample through the wrap
            end_rad = end_rad + 2 * np.pi

        # Ensure we sample enough points along the arc
        angles = np.linspace(start_rad, end_rad, max(n_points, 50))
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        pts = np.column_stack([x, y])
        return pts
    except Exception as e:
        return np.empty((0, 2))


def _sample_spline(spline_entity, n_points: int = 100) -> np.ndarray:
    """Sample a B-spline entity using scipy's B-spline evaluation."""
    try:
        from scipy.interpolate import BSpline

        ctrl_pts = np.array([(float(pt[0]), float(pt[1])) for pt in spline_entity.control_points])
        knots = np.array(spline_entity.knots)
        degree = int(spline_entity.dxf.degree)

        if len(ctrl_pts) < 2 or len(knots) < 2:
            return np.empty((0, 2))

        # Create B-spline objects for x and y
        spl_x = BSpline(knots, ctrl_pts[:, 0], degree, extrapolate=False)
        spl_y = BSpline(knots, ctrl_pts[:, 1], degree, extrapolate=False)

        # Sample across the valid parameter range [knots[degree], knots[-degree-1]]
        u_min = knots[degree]
        u_max = knots[-degree - 1]
        u_sample = np.linspace(u_min, u_max, n_points)

        x = spl_x(u_sample)
        y = spl_y(u_sample)
        return np.column_stack([x, y])
    except ImportError:
        # Fallback: just connect control points
        ctrl_pts = np.array([(float(pt[0]), float(pt[1])) for pt in spline_entity.control_points])
        if len(ctrl_pts) < 2:
            return np.empty((0, 2))
        # Linear interpolation between control points
        pts = []
        for i in range(len(ctrl_pts) - 1):
            p0, p1 = ctrl_pts[i], ctrl_pts[i + 1]
            segment = np.linspace(p0, p1, n_points // (len(ctrl_pts) - 1) + 1)[:-1]
            pts.extend(segment)
        pts.append(ctrl_pts[-1])
        return np.array(pts)
    except Exception:
        return np.empty((0, 2))


def _chain_segments(segments: list) -> Optional[list]:
    """Chain line segments into an ordered polyline, returning list of (x, y) tuples.

    Handles gaps by connecting the closest available segment endpoint.
    """
    if not segments:
        return None

    ordered = [segments[0][0], segments[0][1]]
    remaining = list(segments[1:])
    tol = 1e-3  # 1 mm tolerance for "close enough"

    while remaining:
        last = ordered[-1]
        best_idx = -1
        best_dist = np.inf
        best_reverse = False

        # Find closest segment endpoint
        for i, (a, b) in enumerate(remaining):
            dist_a = np.hypot(last[0] - a[0], last[1] - a[1])
            dist_b = np.hypot(last[0] - b[0], last[1] - b[1])
            if dist_a < best_dist:
                best_dist = dist_a
                best_idx = i
                best_reverse = False
            if dist_b < best_dist:
                best_dist = dist_b
                best_idx = i
                best_reverse = True

        if best_idx >= 0:
            a, b = remaining[best_idx]
            if best_reverse:
                ordered.append(a)
            else:
                ordered.append(b)
            remaining.pop(best_idx)
        else:
            break

    return ordered


def _orient_tip_at_origin(pts: np.ndarray) -> np.ndarray:
    """
    Orient outline so ski runs along +Y from (0, 0).
    Finds the tip (closest point to origin), translates it to (0,0),
    and orients so ski extends in +Y direction.
    """
    # Find the closest point to origin (the tip)
    dists = np.hypot(pts[:, 0], pts[:, 1])
    tip_idx = int(np.argmin(dists))
    tip = pts[tip_idx]

    # Translate so tip is at origin
    pts = pts - tip[np.newaxis, :]

    # Roll so tip is first point
    pts = np.roll(pts, -tip_idx, axis=0)

    # Flip Y if ski extends in negative Y direction
    # (i.e., most points have negative Y or the range is in negative space)
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
    if y_min < -100:  # tail is far in negative direction
        pts[:, 1] = -pts[:, 1]

    # Normalize Y so tip is at y=0
    pts[:, 1] = pts[:, 1] - pts[0, 1]

    return pts


def compute_geometry(outline: np.ndarray, params: SkiParams) -> SkiGeometry:
    """Derive all geometry from planform outline and parameters."""
    geom = SkiGeometry(outline=outline)

    y = outline[:, 1]
    geom.ski_length = float(y.max() - y.min())
    geom.geometric_centre_x = geom.ski_length / 2.0

    # Waist = point of minimum full-ski width (average of left/right half-widths)
    geom.waist_x = _find_waist(outline)
    geom.setback = geom.waist_x - geom.geometric_centre_x

    geom.core_tip_x = params.tip_infill
    geom.core_tail_x = geom.ski_length - params.tail_infill

    centre_uf = geom.waist_x
    geom.underfoot_start_x = centre_uf - params.underfoot_length / 2.0
    geom.underfoot_end_x = centre_uf + params.underfoot_length / 2.0
    geom._underfoot_thickness = params.underfoot_thickness

    geom._thickness_spline_tip, geom._thickness_spline_tail = _build_thickness_splines(geom, params)

    return geom


def _find_waist(outline: np.ndarray) -> float:
    """Return the Y position of minimum ski width in the sidecut region.

    Strategy: sample width along the full length, then restrict the search to
    the region between the two width-peak positions (tip flare and tail flare).
    This avoids the tapered zones near tip and tail where width ramps from zero.
    """
    y_vals = outline[:, 1]
    y_min, y_max = y_vals.min(), y_vals.max()
    sample_y = np.linspace(y_min + 1, y_max - 1, 800)

    widths = np.full(len(sample_y), np.nan)
    for i, sy in enumerate(sample_y):
        xs = _outline_x_at_y(outline, sy)
        if len(xs) >= 2:
            widths[i] = xs.max() - xs.min()

    valid = ~np.isnan(widths)
    if not valid.any():
        return float((y_min + y_max) / 2.0)

    w_valid = widths.copy()
    w_valid[~valid] = 0.0
    w_max = w_valid.max()

    # Find the first and last positions where width exceeds 90% of maximum.
    # The waist lies between these two points (the sidecut region).
    threshold = 0.90 * w_max
    above = np.where(valid & (w_valid >= threshold))[0]
    if len(above) < 2:
        # Fallback: search middle 30–70%
        length = y_max - y_min
        mask = (sample_y >= y_min + 0.30 * length) & (sample_y <= y_min + 0.70 * length)
    else:
        mask = np.zeros(len(sample_y), dtype=bool)
        mask[above[0]: above[-1] + 1] = True
        mask &= valid

    if not mask.any():
        mask = valid

    waist_idx = int(np.nanargmin(np.where(mask, widths, np.nan)))
    return float(sample_y[waist_idx])


def _outline_x_at_y(outline: np.ndarray, y_target: float) -> np.ndarray:
    """Return all X intersections of the outline polyline at a given Y."""
    xs = []
    for i in range(len(outline) - 1):
        y0, y1 = outline[i, 1], outline[i + 1, 1]
        if (y0 <= y_target <= y1) or (y1 <= y_target <= y0):
            if abs(y1 - y0) > 1e-9:
                t = (y_target - y0) / (y1 - y0)
                x = outline[i, 0] + t * (outline[i + 1, 0] - outline[i, 0])
                xs.append(x)
    return np.array(xs)


def _build_thickness_splines(geom: SkiGeometry, params: SkiParams):
    """
    Build two separate cubic splines: one for tip taper, one for tail taper.

    Each spline enforces zero slope at both endpoints:
      Tip spline: core_tip_x → underfoot_start_x (tip_thickness → underfoot_thickness)
      Tail spline: underfoot_end_x → core_tail_x (underfoot_thickness → tail_thickness)

    The underfoot region itself is handled separately as a constant thickness.
    """
    bc_zero = [(1, 0.0), (1, 0.0)]  # zero slope at both ends

    # Tip taper spline
    x_tip = [geom.core_tip_x, geom.underfoot_start_x]
    t_tip = [params.tip_thickness, params.underfoot_thickness]
    spline_tip = CubicSpline(x_tip, t_tip, bc_type=bc_zero)

    # Tail taper spline
    x_tail = [geom.underfoot_end_x, geom.core_tail_x]
    t_tail = [params.underfoot_thickness, params.tail_thickness]
    spline_tail = CubicSpline(x_tail, t_tail, bc_type=bc_zero)

    return spline_tip, spline_tail


def half_widths_at_y(outline: np.ndarray, y_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return left (negative X) and right (positive X) edge at each Y sample."""
    left = np.full_like(y_vals, np.nan)
    right = np.full_like(y_vals, np.nan)
    for i, sy in enumerate(y_vals):
        xs = _outline_x_at_y(outline, sy)
        if len(xs) >= 2:
            left[i] = xs.min()
            right[i] = xs.max()
    return left, right
