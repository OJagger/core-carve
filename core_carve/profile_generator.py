"""G-code generation for core thickness profiling."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class ProfileParams:
    """CNC machine and tool parameters for thickness profiling."""
    tool_diameter: float = 12.0
    spindle_speed: int = 12000
    cutting_feed: float = 800.0
    plunge_feed: float = 200.0
    rapid_feed: float = 1500.0
    roughing_depth_per_pass: float = 2.0
    finishing_depth_of_cut: float = 0.5
    stepover: float = 6.0
    clearance_height: float = 10.0
    direction: str = "along"  # "along" (ski length) or "across" (width)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "ProfileParams":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Move:
    """A single CNC move."""
    x: float
    y: float
    z: float
    is_rapid: bool = False
    feed: float = 0.0


def generate_profile_gcode(
    geom,
    params,
    blank,
    profile_params: ProfileParams,
) -> tuple[str, list[Move]]:
    """
    Generate G-code for core thickness profiling in blank-space coordinates.

    Blank space:
      X = along blank = ski_Y + x_offset  (ski runs along blank X)
      Y = across blank = ski_X + core_y_offset

    Returns:
        (gcode_string, moves_list)  — moves are in blank space for visualization.
    """
    from core_carve.core_blank import MachineOrientation, OriginCorner

    core_positions = blank.get_core_positions(geom, params)
    if not core_positions:
        return "", []

    core_start = geom.core_tip_x - 25.0
    core_end = geom.core_tail_x + 25.0
    core_extent = core_end - core_start

    # Centering offset: same calculation as tab_gcode.py
    x_offset = (blank.length - core_extent) / 2.0 - core_start

    def ski_to_blank(y_ski: float, x_ski: float) -> tuple[float, float]:
        """Convert ski-geometry coords to blank-space coords."""
        return y_ski + x_offset, x_ski

    def transform_to_machine_space(bx: float, by: float) -> tuple[float, float]:
        """Apply machine orientation and origin corner transforms."""
        if blank.machine_orientation == MachineOrientation.Y_AXIS:
            bx, by = by, bx
        if blank.origin_corner == OriginCorner.TOP_LEFT:
            by = blank.width - by
        elif blank.origin_corner == OriginCorner.TOP_RIGHT:
            bx = blank.length - bx
            by = blank.width - by
        elif blank.origin_corner == OriginCorner.BOTTOM_RIGHT:
            bx = blank.length - bx
        return bx, by

    moves: list[Move] = []
    gcode = [
        "G21 G17 G90 G94",
        f"G00 Z{profile_params.clearance_height:.3f}",
        f"S{profile_params.spindle_speed} M03",
    ]

    def rapid(bx, by, bz):
        moves.append(Move(bx, by, bz, is_rapid=True))
        mx, my = transform_to_machine_space(bx, by)
        gcode.append(f"G00 X{mx:.3f} Y{my:.3f} Z{bz:.3f}")

    def feed_move(bx, by, bz, f):
        moves.append(Move(bx, by, bz, is_rapid=False, feed=f))
        mx, my = transform_to_machine_space(bx, by)
        gcode.append(f"G01 X{mx:.3f} Y{my:.3f} Z{bz:.3f} F{f:.1f}")

    # Thickness profile: total depth = blank thickness (simplified flat profile for now)
    total_depth = blank.thickness
    depths = []
    d = profile_params.roughing_depth_per_pass
    while d < total_depth - profile_params.finishing_depth_of_cut:
        depths.append(-d)
        d += profile_params.roughing_depth_per_pass
    depths.append(-total_depth + profile_params.finishing_depth_of_cut)
    depths.append(-total_depth)

    for core_x_along, core_y_across in core_positions:
        # Width of core to cover (use half-widths at waist + sidewall allowance)
        half_w = params.sidewall_width + getattr(params, "sidewall_overlap", 0.0)

        if profile_params.direction == "along":
            # Passes run along the ski (blank X axis), spaced across width (blank Y axis)
            y_ski_samples = np.linspace(core_start, core_end, 300)
            x_ski_passes = np.arange(-half_w, half_w + profile_params.stepover, profile_params.stepover)

            for depth in depths:
                for i, x_ski in enumerate(x_ski_passes):
                    bx_start, by = ski_to_blank(core_start, x_ski + core_y_across)
                    bx_end, _ = ski_to_blank(core_end, x_ski + core_y_across)

                    rapid(bx_start, by, profile_params.clearance_height)
                    feed_move(bx_start, by, depth, profile_params.plunge_feed)

                    # Alternate direction each pass (boustrophedon)
                    ys = y_ski_samples if i % 2 == 0 else y_ski_samples[::-1]
                    for y_ski in ys:
                        bx, _ = ski_to_blank(y_ski, x_ski + core_y_across)
                        feed_move(bx, by, depth, profile_params.cutting_feed)

                    rapid(bx_start if i % 2 == 1 else bx_end, by, profile_params.clearance_height)

        else:
            # Passes run across the ski (blank Y axis), spaced along length (blank X axis)
            x_ski_samples = np.arange(-half_w, half_w + profile_params.stepover / 2, profile_params.stepover / 4)
            y_ski_passes = np.arange(core_start, core_end + profile_params.stepover, profile_params.stepover)

            for depth in depths:
                for i, y_ski in enumerate(y_ski_passes):
                    bx, by_start = ski_to_blank(y_ski, -half_w + core_y_across)
                    _, by_end = ski_to_blank(y_ski, half_w + core_y_across)

                    rapid(bx, by_start, profile_params.clearance_height)
                    feed_move(bx, by_start, depth, profile_params.plunge_feed)

                    xs = x_ski_samples if i % 2 == 0 else x_ski_samples[::-1]
                    for x_ski in xs:
                        _, by = ski_to_blank(y_ski, x_ski + core_y_across)
                        feed_move(bx, by, depth, profile_params.cutting_feed)

                    rapid(bx, by_start if i % 2 == 1 else by_end, profile_params.clearance_height)

    gcode += [f"G00 Z{profile_params.clearance_height:.3f}", "M05", "M30"]
    return "\n".join(gcode), moves
